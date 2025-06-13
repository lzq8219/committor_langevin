import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from muller_potential import MullerPotential
import matplotlib.pyplot as plt
import torch.nn.functional as F
from committor_langevin.src.hist import hist_reweight
from utils import read_COLVAR

# Define a simple neural network to represent the function v


class ResidualBlock(nn.Module):
    def __init__(self, inputs, stride=1):
        super(ResidualBlock, self).__init__()

        # Shortcut connection
        self.l1 = nn.Linear(inputs, inputs)
        self.l2 = nn.Linear(inputs, inputs)

        self.aticv = nn.ReLU()

    def forward(self, x):
        y = self.l1(x)
        y = self.aticv(y)**3
        y = self.l2(y)
        y = self.aticv(y)**3 + x
        return y


class GaussianCDFActivation(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        """
        Initialize the Gaussian CDF activation function.

        Parameters:
        - mean: The mean of the Gaussian distribution (default is 0).
        - std: The standard deviation of the Gaussian distribution (default is 1).
        """
        super(GaussianCDFActivation, self).__init__()

    def forward(self, x):
        return (1 + torch.erf(x)) / 2


class FunctionModel(nn.Module):
    def __init__(self, layer_sizes, activation='linear'):
        super(FunctionModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            # self.layers.append(ResidualBlock(layer_sizes[i]))

            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            if i < len(layer_sizes) - 2:  # No activation after the last layer
                self.layers.append(nn.Tanh())

        if activation == 'sigmoid':
            self.layers.append(nn.Sigmoid())
        elif activation == 'tanh':
            self.layers.append(nn.Tanh())
        elif activation == 'gcdf':
            self.layers.append(GaussianCDFActivation())
        elif activation == 'relu':
            self.layers.append(nn.ReLU())
        elif activation == 'softplus':
            self.layers.append(nn.Softplus())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Initialize weights and biases to zero
                nn.init.xavier_normal_(layer.weight)


# Function to compute the integral I[v]


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

# Load the model


def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
    print(f"Model loaded from {file_path}")


class Bacl_bolSolver:
    def __init__(self, layer_sizes, kbt=1, activation='linear',
                 learning_rate=0.01, num_epochs=1000, device=None):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initialize model and optimizer
        self.model = FunctionModel(layer_sizes, activation)

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate)

    def compute_gradient(self, x_values, create_graph=True):
        x_values.requires_grad_(True)  # Enable gradient tracking
        v_values = self.model(x_values)

        # Compute gradients
        gradients = torch.autograd.grad(outputs=v_values, inputs=x_values,
                                        grad_outputs=torch.ones_like(v_values),
                                        create_graph=create_graph, retain_graph=True)[0]
        return gradients

    def compute_integral(self, x_values, weight, f_values,
                         create_graph=True, cumulative=None):
        x_values.requires_grad_(True)  # Enable gradient tracking
        ttt = 10**0
        v_values = self.model(x_values) * ttt

        # Compute gradients
        gradients = torch.autograd.grad(outputs=v_values, inputs=x_values,
                                        grad_outputs=torch.ones_like(v_values),
                                        create_graph=create_graph, retain_graph=True)[0]
        grad_magnitude_squared = torch.sum(
            gradients**2, dim=1, keepdim=True)  # |∇v|^2

        if cumulative is None:
            integral = torch.sum(
                weight * (grad_magnitude_squared.squeeze() / 2 + ttt * f_values * v_values.squeeze())) / torch.sum(weight)
        else:
            integral = torch.sum(
                weight * (grad_magnitude_squared.squeeze() / 2 - cumulative + ttt * f_values * v_values.squeeze())) / torch.sum(weight)

        '''
        integral = torch.sum(grad_magnitude_squared +
                             v_values) / weight.shape[0] * 100
        '''
        return integral

    def create_batches(self, data, batch_size):
        indices = torch.randperm(data[0].size(0))
        for t in data:
            t = t[indices]

        # Create mini-batches
        mini_batches = []
        for i in range(0, data[0].size(0), batch_size):
            mini_batch = [t[i:i + batch_size] for t in data]
            mini_batches.append(mini_batch)

        return mini_batches

    def train(self, x_values, weight, f_values, xb_values,
              b_values, batchsize=None, l_b=1, l_func=1, num_epochs=None, optimizer=None, cp=10, l2=1e-3, cumulative=None):
        # Optimization loop
        loss_list = []
        functional_loss_list = []
        boundary_loss_list = []

        N = xb_values.shape[0]
        if batchsize is None:
            batchsize = x_values.shape[0]
        if num_epochs is None:
            num_epochs = self.num_epochs
        if optimizer is None:
            optimizer = self.optimizer
        for epoch in range(num_epochs):
            if cumulative is None:
                batches = self.create_batches(
                    [x_values, weight, f_values], batch_size=batchsize)
            else:
                batches = self.create_batches(
                    [x_values, weight, f_values, cumulative], batch_size=batchsize)

            for batch in batches:
                x = batch[0]
                w = batch[1]
                f = batch[2]
                if cumulative is not None:
                    c = batch[3]
                else:
                    c = None
                self.optimizer.zero_grad()

                # Compute the integral I[v]
                integral_value = self.compute_integral(x, w, f, cumulative=c)

                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param) ** 2

                # Compute the loss (we want to minimize I[v])

                b_loss = torch.sum(
                    (self.model(xb_values).squeeze() - b_values)**2) / N
                loss = integral_value * l_func + l_b * b_loss + l2 * l2_reg
                # Backpropagation
                loss.backward()
                optimizer.step()
            # print(epoch)
            if epoch % cp == 0:
                print(
                    f'Epoch {epoch}, Loss: {loss.item()}, functional loss:{integral_value.item()}, boundary loss:{b_loss.item()}')
                loss_list.append(loss.item())
                functional_loss_list.append(integral_value.item())
                boundary_loss_list.append(b_loss.item())
        return loss_list, functional_loss_list, boundary_loss_list

    def evaluate(self, x_values, weight):
        pass


class Committor:
    def __init__(self, layer_sizes, activation='linear',
                 learning_rate=0.01, num_epochs=1000, device=None):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initialize model and optimizer
        self.model = FunctionModel(layer_sizes, activation)

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate)

    def compute_gradient(self, x_values, create_graph=True):
        x_values.requires_grad_(True)  # Enable gradient tracking
        v_values = self.model(x_values)

        # Compute gradients
        gradients = torch.autograd.grad(outputs=v_values, inputs=x_values,
                                        grad_outputs=torch.ones_like(v_values),
                                        create_graph=create_graph, retain_graph=True)[0]
        return gradients

    def compute_integral(self, x_values, weight, f_values,
                         create_graph=True, cumulative=None):
        x_values.requires_grad_(True)  # Enable gradient tracking
        ttt = 10**0
        v_values = (torch.erf(self.model(x_values)) + 1) / 2 * ttt

        # Compute gradients
        gradients = torch.autograd.grad(outputs=v_values, inputs=x_values,
                                        grad_outputs=torch.ones_like(v_values),
                                        create_graph=create_graph, retain_graph=True)[0]
        grad_magnitude_squared = torch.sum(
            gradients**2, dim=1, keepdim=True)  # |∇v|^2

        if cumulative is None:
            integral = torch.sum(
                weight * (grad_magnitude_squared.squeeze() / 2 * kbt + ttt * f_values * v_values.squeeze())) / torch.sum(weight)
        else:
            integral = torch.sum(
                weight * (grad_magnitude_squared.squeeze() / 2 * kbt - cumulative + ttt * f_values * v_values.squeeze())) / torch.sum(weight)

        '''
        integral = torch.sum(grad_magnitude_squared +
                             v_values) / weight.shape[0] * 100
        '''
        return integral

    def create_batches(self, data, batch_size):
        if batch_size == data[0].size(0):
            mini_batch = data
        else:
            indices = torch.randperm(data[0].size(0))
            for t in data:
                t = t[indices]

            # Create mini-batches
            mini_batches = []
            for i in range(0, data[0].size(0), batch_size):
                mini_batch = [t[i:i + batch_size] for t in data]
                mini_batches.append(mini_batch)
        return mini_batches

    def train(self, x_values, weight, f_values, xb_values,
              b_values, batchsize=None, l_b=1, l_func=1, num_epochs=None, optimizer=None, cp=10, l2=1e-6, cumulative=None):
        # Optimization loop
        loss_list = []
        functional_loss_list = []
        boundary_loss_list = []
        params = [p for p in self.model.parameters()]

        N = xb_values.shape[0]
        if batchsize is None:
            batchsize = x_values.shape[0]
        if num_epochs is None:
            num_epochs = self.num_epochs
        if optimizer is None:
            optimizer = self.optimizer
        for epoch in range(num_epochs):
            if cumulative is None:
                batches = self.create_batches(
                    [x_values, weight, f_values], batch_size=batchsize)
            else:
                batches = self.create_batches(
                    [x_values, weight, f_values, cumulative], batch_size=batchsize)
            for batch in batches:
                x = batch[0]
                w = batch[1]
                f = batch[2]
                if cumulative is not None:
                    c = batch[3]
                else:
                    c = None
                self.optimizer.zero_grad()

                # Compute the integral I[v]
                integral_value = self.compute_integral(x, w, f, cumulative=c)

                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param) ** 2

                # Compute the loss (we want to minimize I[v])
                b_loss = torch.sum(
                    (torch.erf(self.model(xb_values)).squeeze() + 0.5 - b_values)**2) / N
                loss = integral_value * l_func + l_b * b_loss + l2 * l2_reg

                # Backpropagation
                loss.backward(inputs=params)
                optimizer.step()
            # print(epoch)
            if epoch % cp == 0:
                integral_value = 0
                for batch in batches:
                    x = batch[0]
                    w = batch[1]
                    f = batch[2]
                    if cumulative is not None:
                        c = batch[3]
                    else:
                        c = None
                    # Compute the integral I[v]
                    integral_value += self.compute_integral(
                        x, w, f, cumulative=c, create_graph=False)
                loss = l_func * integral_value + l_b * b_loss
                print(
                    f'Epoch {epoch}, Loss: {loss.item()}, functional loss:{integral_value.item()}, boundary loss:{b_loss.item()}')
                loss_list.append(loss.item())
                functional_loss_list.append(integral_value.item())
                boundary_loss_list.append(b_loss.item())
        return loss_list, functional_loss_list, boundary_loss_list

    def evaluate(self, x_values, weight):
        pass


def compute_hessian(v, x):
    # Ensure v is a vector and x is a matrix
    assert v.dim() == 1, "v must be a 1D tensor (shape: (n,))"
    assert x.dim() == 2, "x must be a 2D tensor (shape: (n, m))"

    n = v.size(0)
    m = x.size(1)

    # Initialize the Hessian tensor
    hessian = torch.zeros((n, m, m), dtype=x.dtype, device=x.device)
    dv = torch.autograd.grad(outputs=v, inputs=x,
                             grad_outputs=torch.ones_like(v),
                             create_graph=True, retain_graph=True)[0]
    # Compute the Hessian for each component of v
    for j in range(m):
        print(dv[:, j].shape)
        hessian[:, :, j] = torch.autograd.grad(outputs=dv[:, j], inputs=x,
                                               grad_outputs=torch.ones_like(
                                                   dv[:, j]),
                                               create_graph=False, retain_graph=True)[0]

    return hessian


def LBFGS(xint, device, loss_fn, num_steps, lr=1):
    x = xint  # Start at x=0
    x.to(device).requires_grad_(True)

    # Define the optimizer
    optimizer = torch.optim.Adam([x], lr=1)

    # Optimization loop

    # Run the optimization
    for i in range(num_steps):  # Number of iterations
        optimizer.zero_grad()  # Zero the gradients
        # We minimize the negative of f(x) to maximize f(x)
        loss = loss_fn(x)
        loss.backward()
        optimizer.step()
        print(x)

    return x


def compute_expression(x, du, v, kbt):
    # Ensure u and v are 2D tensors (e.g., images)
    hessain = compute_hessian(v.squeeze(), x)
    n = v.size(0)
    m = x.size(1)

    # Compute Laplacian of v
    l = torch.zeros((n,), dtype=x.dtype, device=x.device)
    for i in range(n):
        for j in range(m):
            l[i] += hessain[i, j, j]
    dv = torch.autograd.grad(outputs=v, inputs=x,
                             grad_outputs=torch.ones_like(v),
                             create_graph=False, retain_graph=False)[0]
    dot_product = torch.sum(du * dv, dim=1)
    # Compute the final expression
    result = -dot_product + kbt * l

    return result


if __name__ == "__main__":
    # Create an instance of the Müller potential
    muller = MullerPotential()

    # Define input array (e.g., coordinates of particles)

    n = 100
    nb = 100
    kbt = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './model/'

    # uniform mesh
    '''
    x = np.linspace(-1.5, 1.2, n)

    y = np.linspace(-0.2, 2, n)

    X, Y = np.meshgrid(x, y)

    XX = np.reshape(X, -1)
    YY = np.reshape(Y, -1)
    XXX = np.reshape(XX, X.shape)
    x_values = np.array([XX, YY]).T
    '''

    # test with other problem
    '''
    x1 = np.array([x, np.ones_like(x) * (-0.2)]).T
    x2 = np.array([x, np.ones_like(x) * 2]).T
    y1 = np.array([np.ones_like(y) * (-1.5), y]).T
    y2 = np.array([np.ones_like(y) * 1.2, y]).T
    x3 = np.array([np.linspace(0, 1.2, n), np.ones(n)]).T
    xbb = np.concatenate((x1, x2, y1, y2, x3), axis=0)
    vbb = np.zeros(xbb.shape[0])
    '''

    # data_A = read_COLVAR('simulation/A/COLVAR_25.00')
    # data_B = read_COLVAR('simulation/B/COLVAR_25.00')
    data_A = read_COLVAR('muller_25_A_5e5.txt')
    data_B = read_COLVAR('muller_25_A_5e5.txt')
    stride = 20
    x_values_A = data_A
    x_values_B = data_B
    '''
    x_values_A = data_A[::stride, 1:3]
    U_A = data_A[::stride, 3]
    U_B = data_B[::stride, 3]
    Umin = np.min([U_A, U_B])
    w_A = np.sum(np.exp(-(U_A - Umin) / kbt))
    x_values_B = data_B[::stride, 1:3]
    U_B = data_B[::stride, 3]
    w_B = np.sum(np.exp(-(U_B - Umin) / kbt))
    wsum = w_A + w_B
    w_A = w_A / wsum
    w_B = w_B / wsum
    print(w_A, w_B)
    '''
    w_A = 1 / 2
    w_B = 1 / 2

    x_values = np.concatenate((x_values_A, x_values_B), axis=0)
    w = np.concatenate((w_A *
                        np.ones_like(x_values_A[:, 0]), w_B *
                        np.ones_like(x_values_B[:, 0])), axis=0)

    # x_values = np.loadtxt('muller_0.15.txt', dtype=np.float32)
    # x_values = np.random.uniform(size=(n * n, 2)) * \
    #    np.array([2.7, 2.2]) - np.array([1.5, 0.2])

    w = w[~muller.in_a(x_values)]
    x_values = x_values[~muller.in_a(x_values)]
    w = w[~muller.in_b(x_values)]
    x_values = x_values[~muller.in_b(x_values)]

    print(x_values.shape)
    center = [-0.75, 0.6]
    l = np.sum((x_values - center)**2, axis=1) < 0.5**2

    '''
    U = muller.potential(x_values)

    w = np.exp(- (U - np.min(U)) / kbt)
    w = w / np.sum(w)
    '''
    # w = np.ones_like(x_values[:, 0])

    # show the training set(invariant distribution)
    '''
    ngrid = 400
    grid = np.linspace(-2, 2, ngrid)
    y, x = np.meshgrid(grid, grid)
    y = y.flatten()
    x = x.flatten()
    g = np.array([x, y]).T
    # filename='simulation/long/COLVAR'
    data = x_values

    # fes=calculateFES_multi(df,grid,16)
    nstart = 0
    h = hist_reweight(data, w, -2, 2, -2, 2, ngrid)

    h = h.flatten()
    cc = np.log(h[h > 0])
    thread = -20
    cc[cc < thread] = thread
    plt.scatter(g[:, 0][h > 0], g[:, 1][h > 0],
                cmap='turbo', c=cc, s=1)
    plt.xlabel('$\\phi$')
    plt.ylabel('$\\psi$')
    plt.colorbar(label='FES')
    plt.show()
    '''
    '''

    num_points = 2**16
    x = np.random.normal(size=(num_points, 2))
    center = np.array([-0.5, 0.6])
    r = 0.5
    x_values = x * r + center
    x_values = x_values[~muller.in_a(x_values)]
    x_values = x_values[~muller.in_b(x_values)]
    '''

    '''
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_values[:, 0], x_values[:, 1], alpha=0.1)
    plt.title('2D Contour Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # Add color bar

    # Add labels and title

    plt.show()
    '''

    '''
    U = muller.potential(x_values)

    w = np.exp(- (U - np.min(U)) / kbt +
               np.sum((x_values - center)**2, axis=1) / 2 / r**2)
    t = - (U - np.min(U)) / kbt +\
        np.sum((x_values - center)**2, axis=1) / 2 / r**2
    t[t < -20] = -20
    plt.scatter(x_values[:, 0], x_values[:, 1], c=t, cmap='cool')
    plt.colorbar()
    plt.show()
    w = w / np.sum(w) * 2 * np.pi * r**2
    '''

    f = -np.ones((x_values.shape[0],)) * 1
    # f = -np.ones((x_values.shape[0],)) * 1

    x_a = muller.points_on_a_boundary(nb)
    x_aa = muller.points_in_a(nb)
    x_b = muller.points_on_b_boundary(nb)
    x_bb = muller.points_in_b(nb)
    q_a = 1 - np.ones((nb,))
    q_aa = q_a.copy()
    q_b = + np.zeros((nb,))
    q_bb = q_b.copy()

    x_values = torch.from_numpy(x_values.astype(np.float32))
    f_values = torch.from_numpy(f.astype(np.float32))
    weight = torch.from_numpy(w.astype(np.float32))

    x_values = x_values.to(device)
    f_values = f_values.to(device)
    weight = weight.to(device)

    xb = np.concatenate((x_a, x_aa, x_b, x_bb), axis=0)

    b_values = np.concatenate((q_a, q_aa, q_b, q_bb), axis=0)
    xb_values = torch.from_numpy(xb.astype(np.float32))
    xb_values = xb_values.to(device)
    b_values = torch.from_numpy(b_values.astype(np.float32))
    b_values = b_values.to(device)

    # training

    layers = [2, 16, 32, 16, 1]
    # layers = [2, 4, 8, 4, 1]
    activ = 'linear'

    q = Bacl_bolSolver(
        layer_sizes=layers,
        activation=activ,
        learning_rate=0.001,
        kbt=kbt)
    ls = []
    fls = []
    bls = []
    cp = 200
    num_epochs = 60000
    for lr in [10**(-i) for i in range(4, 5)]:

        q.model.initialize_weights()
        opt = optim.Adam(q.model.parameters(), lr=lr)
        l, fl, bl = q.train(
            x_values,
            weight,
            f_values,
            xb_values,
            b_values,
            l_b=1e5,
            l_func=1,
            batchsize=None,
            num_epochs=num_epochs,
            optimizer=opt,
            cp=cp,
            l2=0)

        epochs = np.arange(len(l)) * cp
        fig, axs = plt.subplots(1, 3, figsize=(15, 7.5))
        l_list = [l, fl, bl]
        ys = ['loss', 'functional loss', 'boundary loss']
        nstart = int(num_epochs / cp * 0.2)

        for i in range(3):
            axs[i].plot(epochs[nstart:], l_list[i][nstart:],
                        marker='o', color='b', label=ys[i])
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel(ys[i])
            axs[i].legend()
            axs[i].grid()
            # axs[i].set_yscale('log')

        plt.savefig(f'./pic/T=25/loss_m_{lr:1f}.png', dpi=300)

        plt.clf()

        model_filename = model_path + f'T=25/m_lr={lr:1f}.pth'
        save_model(q.model, model_filename)

    # training over

        x = np.linspace(-1.5, 1.2, n)
        y = np.linspace(-0.2, 2, n)
        X, Y = np.meshgrid(x, y)

        XX = np.reshape(X, -1)
        YY = np.reshape(Y, -1)
        model = FunctionModel(layer_sizes=layers, activation=activ)
        model_filename = model_path + f'T=25/m_lr={lr:1f}.pth'
        load_model(model, model_filename)
        model.to(device)
        x_values_1 = np.array([XX, YY]).T
        dU = muller.gradient(x_values_1)
        x_values_1 = torch.from_numpy(x_values_1.astype(np.float32))
        dU = torch.from_numpy(dU.astype(np.float32))

        x_values_1 = x_values_1.to(device)
        x_values_1.requires_grad_(True)
        dU = dU.to(device)
        z = model(x_values_1)

        Lv = compute_expression(x_values_1, dU, z, kbt)

        print(Lv.mean(), Lv.median())

        x = x_values_1.to('cpu').detach().numpy()
        z = z.to('cpu').detach().numpy()
        zm = np.mean(z)
        # z[z > zm] = zm
        Z = np.reshape(z, X.shape)

        Lv = Lv.to('cpu').detach().numpy()
        plt.scatter(x[:, 0], x[:, 1], c=Lv, cmap='cool')
        plt.colorbar()
        plt.savefig(f'pic/T=25/m_pinn_loss_{lr:1f}.png')
        plt.clf()
        u = np.reshape(muller.potential(np.array([XX, YY]).T), X.shape)
        u[u > 0] = 0

        # Draw potential

        plt.figure(figsize=(8, 6))
        contour = plt.contour(
            X,
            Y,
            u, levels=20,
            cmap='Spectral')  # 20 contour levels
        plt.colorbar(contour)  # Add a colorbar to indicate the scale
        plt.title('2D Contour Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)

        # Add color bar

        # Add labels and title
        Z[Z > 500] = 500
        Z[Z < 0] = 0
        contour1 = plt.contour(X, Y, Z, levels=20, cmap='turbo')
        plt.colorbar(contour1)
        plt.savefig(f'pic/T=25/m_{lr:1f}.png')
        plt.clf()

        data = np.loadtxt('./model/T=25/m_fd.txt')
        x_values_1 = data[:, 0:2]
        u1 = muller.potential(x_values_1)
        # l = ~((~muller.in_a(x_values_1)) * (~muller.in_b(x_values_1)))
        x_values_1 = x_values_1
        u1 = u1

        w1 = np.exp(-(u1 - np.min(u1)) / kbt)
        w1 = w1 / np.sum(w1)
        m = data[:, 2]
        m = m
        x_values_1 = torch.from_numpy(x_values_1.astype(np.float32))

        x_values_1 = x_values_1.to(device)
        z = model(x_values_1)
        z = z.squeeze().to('cpu').detach().numpy()
        print(z)

        res = m - z

        threshold = -20

        error = np.sum(res**2 * w1) / np.sum(m**2 * w1)
        print(
            f'relative error: {error}, m l2: {np.sum(m**2*w1)}, res l2: {np.sum(res**2*w1)}')

        plt.figure(figsize=(8, 6))
        contour = plt.contour(
            X,
            Y,
            u, levels=20,
            cmap='Spectral')  # 20 contour levels
        plt.colorbar(contour)  # Add a colorbar to indicate the scale
        plt.title('2D Contour Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.scatter(data[:, 0], data[:, 1], c=res, cmap='turbo')
        plt.colorbar()
        plt.savefig(f'pic/T=25/m_res_{lr:1f}.png')

        plt.clf()
