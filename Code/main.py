import numpy as np
import torch
from back_kol_solver import load_model, FunctionModel, BackKolSolver, Committor, LBFGS
from muller_potential import MullerPotential
import matplotlib.pyplot as plt
import torch.optim as optim
from langevin_simulation import ul_simulation
from hist import hist_reweight


def grad(x0: np.ndarray, model, device, tempered):
    x = torch.from_numpy(x0).to(device)
    x.requires_grad_(True)
    y = model(x)
    y = (torch.erf(y) + 1) / 2
    gradients = torch.autograd.grad(outputs=y, inputs=x,
                                    grad_outputs=torch.ones_like(y),
                                    create_graph=True, retain_graph=True)[0]
    dv2 = (torch.sum(gradients**2) + 1e-10)**tempered
    g = torch.autograd.grad(outputs=dv2, inputs=x,
                            grad_outputs=torch.ones_like(dv2),
                            create_graph=False, retain_graph=False)[0]
    g = g.to('cpu').detach().numpy()
    return g


# Define the function
layers = [2, 32, 32, 1]
activ = 'gcdf'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


muller = MullerPotential()

n = 200
nb = 300
kbt = 1 / 0.15
x = np.linspace(-1.5, 1.2, n)

y = np.linspace(-0.2, 2, n)

X, Y = np.meshgrid(x, y)

XX = np.reshape(X, -1)
YY = np.reshape(Y, -1)

filename_A = 'muller_0.15_A_1e4.txt'
filename_B = 'muller_0.15_B_1e4.txt'
x_values_A = np.loadtxt(filename_A)
x_values_B = np.loadtxt(filename_B)
w_A = np.sum(np.exp(-muller.potential(x_values_A) / kbt))
w_B = np.sum(np.exp(-muller.potential(x_values_B) / kbt))
print(w_A, w_B)

x_values = np.concatenate((x_values_A, x_values_B), axis=0)
w = np.concatenate((w_A * np.ones(x_values_A.shape[0]),
                   w_B * np.ones(x_values_B.shape[0])), axis=0)
w = w / np.sum(w) * x_values.shape[0]
f = np.zeros_like(w)

x_a = muller.points_on_a_boundary(nb)
x_aa = muller.points_in_a(nb)
x_b = muller.points_on_b_boundary(nb)
x_bb = muller.points_in_b(nb)
q_a = np.ones((nb,))
q_aa = q_a.copy()
q_b = + np.zeros((nb,))
q_bb = q_b.copy()

xb = np.concatenate((x_a, x_aa, x_b, x_bb), axis=0)
b_values = np.concatenate((q_a, q_aa, q_b, q_bb), axis=0)
xb_values = torch.from_numpy(xb.astype(np.float32))
xb_values = xb_values.to(device)
b_values = torch.from_numpy(b_values.astype(np.float32))
b_values = b_values.to(device)
f_values = torch.from_numpy(f.astype(np.float32))
f_values = f_values.to(device)
qlist = []
zlist = []
dq2_tempered_list = []
lambda_list = []
tempered = 1
cum = None

for itr in range(5):
    print(f'itr{itr}:')
    x_values = torch.from_numpy(x_values.astype(np.float32))

    weight = torch.from_numpy(w.astype(np.float32))
    f_values = torch.from_numpy(f.astype(np.float32))
    f_values = f_values.to(device)

    x_values = x_values.to(device)

    weight = weight.to(device)

    qlist.append(Committor(
        layer_sizes=layers,
        activation='linear'))

    ls = []
    fls = []
    bls = []
    cp = 100
    num_epochs = 1000
    print('training')
    for lr in [10**(-i) for i in range(2, 4)]:
        opt = optim.Adam(qlist[-1].model.parameters(), lr=lr)
        l, fl, bl = qlist[-1].train(
            x_values,
            weight,
            f_values,
            xb_values,
            b_values,
            kk=1e3 * 5,
            kkk=1,
            batchsize=2**14,
            num_epochs=num_epochs,
            optimizer=opt,
            cp=cp,
            l2=1e-8,
            cumulative=cum)
        ls += l
        fls += fl
        bls += bl

    muller = MullerPotential()

    x_values = np.array([XX, YY]).T
    x_values = torch.from_numpy(x_values.astype(np.float32))
    x_values = x_values.to(device)
    x_values.requires_grad_(True)
    z = qlist[-1].model(x_values)
    q = (torch.erf(z) + 1) / 2

    def loss_fn(x): return (qlist[-1].model(x) - 0)**2
    xmax = LBFGS(xint=torch.tensor([0.,
                                    0.],
                                   device=device,
                                   requires_grad=True),
                 device=device,
                 loss_fn=loss_fn,
                 num_steps=100, lr=0.001)

    zzz = qlist[-1].model(xmax)
    ggg = torch.autograd.grad(outputs=zzz, inputs=xmax,
                              grad_outputs=torch.ones_like(zzz),
                              create_graph=False, retain_graph=False)[0]
    torch.sum(ggg**2) / np.pi

    gradients = torch.autograd.grad(outputs=q, inputs=x_values,
                                    grad_outputs=torch.ones_like(z),
                                    create_graph=False, retain_graph=False)[0]
    dq2 = (torch.sum(gradients**2, dim=1, keepdim=False) + 1e-10)**tempered
    dq2 = dq2.to('cpu').detach().numpy()
    z = z.to('cpu').detach().numpy()
    dq2_tempered_list.append(dq2)

    u = np.reshape(muller.potential(np.array([XX, YY]).T), X.shape)

    vbias = np.zeros_like(u)
    vv = np.zeros((XX.shape[0],))
    for dq2 in dq2_tempered_list:

        DQ2 = np.reshape(dq2, X.shape)
        vbias += DQ2
        vv += dq2
        # ww = np.exp(-u / kbt) / np.sum(np.exp(u))
        # w = ww.reshape(-1)

    # Draw potential

    t = 0.01
    plt.figure(figsize=(8, 6))
    uuu = (u - t * kbt * vbias) / kbt
    uuu[uuu > 0] = 0
    contour = plt.contour(
        X,
        Y,
        uuu, levels=20,
        cmap='Spectral')  # 20 contour levels
    scatter = plt.scatter(XX, YY, c=vv, cmap='Reds')
    plt.colorbar(contour)  # Add a colorbar to indicate the scale
    plt.colorbar(scatter)
    plt.title(f'T={t},itr{itr}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.savefig(f'pic\\itr_try\\{itr}\\T={t}_c.png')
    plt.clf()

    plt.figure(figsize=(8, 6))
    uuu = u / kbt
    uuu[uuu > 0] = 0
    contour = plt.contour(
        X,
        Y,
        uuu, levels=20,
        cmap='Spectral')  # 20 contour levels
    scatter = plt.scatter(XX, YY, c=z, cmap='Reds')
    plt.colorbar(contour)  # Add a colorbar to indicate the scale
    plt.colorbar(scatter)
    plt.title(f'T={t},itr{itr}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.savefig(f'pic\\itr_try\\{itr}\\T={t}_z.png')
    plt.clf()

    # Add color bar

    # Add labels and title

    '''
    muller = MullerPotential()
    q_truth.to(device)
    x_values = np.array([XX, YY]).T
    dU = muller.gradient(x_values)
    x_values = torch.from_numpy(x_values.astype(np.float32))
    dU = torch.from_numpy(dU.astype(np.float32))

    x_values = x_values.to(device)
    x_values.requires_grad_(True)
    dU = dU.to(device)
    z = q_truth(x_values)

    gradients = torch.autograd.grad(outputs=z, inputs=x_values,
                                    grad_outputs=torch.ones_like(z),
                                    create_graph=False, retain_graph=False)[0]
    dq2 = torch.sum(gradients**2, dim=1, keepdim=False)
    dq2 = dq2.to('cpu').detach().numpy()


    x = x_values.to('cpu').detach().numpy()
    z = z.to('cpu').detach().numpy()
    zm = np.mean(z)
    # z[z > zm] = zm
    Z = np.reshape(z, X.shape)


    u = np.reshape(muller.potential(np.array([XX, YY]).T), X.shape)
    u[u > 0] = 0
    DQ2 = np.reshape(dq2, X.shape)

    # Draw potential

    for t in [10**(-i) for i in range(-2, 2)]:
        plt.figure(figsize=(8, 6))
        contour = plt.contour(
            X,
            Y,
            u - t * DQ2, levels=20,
            cmap='Spectral')  # 20 contour levels
        scatter = plt.scatter(XX, YY, c=dq2, cmap='Reds')
        plt.colorbar(contour)  # Add a colorbar to indicate the scale
        plt.colorbar(scatter)
        plt.title(f'T={t},q_truth')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)

    plt.show()
    '''

    # Add color bar

    # Add labels and title

    def grad_fn(x): return muller.gradient(x) - t * kbt * \
        np.sum([grad(x, q.model.to(torch.device('cpu')),
               torch.device('cpu'), tempered) for q in qlist], axis=0)

    nstep = int(1e4)
    x0_A = muller.c_a()
    x0_A = x0_A.astype(np.float32)
    print(f'itr{itr}: sampling A')
    xs_A, vs_A = ul_simulation(grad_fn, xdim=2, tstep=0.005, kbt=kbt, xinit=x0_A,
                               nstep=nstep, stride_print=False, stride=1)

    x0_B = muller.c_b()
    x0_B = x0_B.astype(np.float32)
    print(f'itr{itr}: sampling B')
    xs_B, vs_B = ul_simulation(grad_fn, xdim=2, tstep=0.005, kbt=kbt, xinit=x0_B,
                               nstep=nstep, stride_print=False, stride=1)

    p_A = muller.potential(xs_A)
    p_B = muller.potential(xs_B)

    for q in qlist:
        q.model.to(device)
        xxx_A = torch.from_numpy(xs_A).to(device)
        xxx_A.requires_grad_(True)
        z_A = q.model(xxx_A)
        g_A = gradients = torch.autograd.grad(outputs=z_A, inputs=xxx_A,
                                              grad_outputs=torch.ones_like(
                                                  z_A),
                                              create_graph=False, retain_graph=False)[0]
        dv2 = (torch.sum(gradients**2, dim=1) + 1e-30)**tempered
        p_A = p_A - t * kbt * dv2.to('cpu').detach().numpy()

        xxx_B = torch.from_numpy(xs_B).to(device)
        xxx_B.requires_grad_(True)
        z_B = q.model(xxx_B)
        g_A = torch.autograd.grad(outputs=z_B, inputs=xxx_B,
                                  grad_outputs=torch.ones_like(
                                      z_B),
                                  create_graph=False, retain_graph=False)[0]
        dv2 = (torch.sum(gradients**2, dim=1) + 1e-30)**tempered
        p_B = p_B - t * kbt * dv2.to('cpu').detach().numpy()

    w_A = np.sum(np.exp(-p_A / kbt))
    w_B = np.sum(np.exp(-p_B / kbt))

    ngrid = 400
    grid = np.linspace(-2, 2, ngrid)
    y, x = np.meshgrid(grid, grid)
    y = y.flatten()
    x = x.flatten()
    g = np.array([x, y]).T
    # filename='simulation/long/COLVAR'
    data = np.concatenate((xs_A, xs_B), axis=0)
    x_values = np.concatenate((x_values_A, x_values_B), axis=0)

    # fes=calculateFES_multi(df,grid,16)
    nstart = 0
    w = np.concatenate((w_A * np.ones(xs_A.shape[0]),
                        w_B * np.ones(xs_B.shape[0])), axis=0)
    w = w / np.sum(w) * w.shape[0]
    h = hist_reweight(data, w, -2, 2, -2, 2, ngrid)
    # hh = hist_reweight(x_values, w, -2, 2, -2, 2, ngrid)

    h = h.flatten()
    # hh = hh.flatten()
    cc = np.log(h[h > 0])
    # ccc = np.log(hh[hh > 0])
    # thread = -20
    # cc[cc < thread] = thread
    plt.scatter(g[:, 0][h > 0], g[:, 1][h > 0],
                cmap='turbo', c=cc, s=1)
    plt.xlabel('$\\phi$')
    plt.ylabel('$\\psi$')
    plt.colorbar(label='biased simulation')
    plt.savefig(f'pic\\itr_try\\{itr}\\bias_simulation.png')
    plt.clf()

    x_values = data
    f = np.zeros_like(w)
