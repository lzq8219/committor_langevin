import numpy as np
import matplotlib.pyplot as plt
import numba
import torch


class MullerPotential:
    def __init__(self, temp: float = 1.):
        self.temp = temp
        pass

    def potential(self, x):
        """
        Calculate the Müller potential for a given input array x.

        Parameters:
        x (ndarray): Input array of shape (n, 2) where n is the number of particles.

        Returns:
        ndarray: The Müller potential evaluated at x.
        """
        # Extract coordinates
        x0 = np.array([1, 0, -0.5, -1])
        y0 = np.array([0, 0.5, 1.5, 1])

        a = np.array([-1, -1, -6.5, 0.7])
        b = np.array([0, 0, 11, 0.6])
        c = np.array([-10, -10, -6.5, 0.7])
        A = np.array([-200, -100, -170, 15])

        # Calculate potential energy

        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                PotEn = np.zeros_like(x[:, 0])
                for k in range(4):
                    a1 = a[k] * (x[:, 0] - x0[k]) ** 2
                    b1 = b[k] * (x[:, 0] - x0[k]) * (x[:, 1] - y0[k])
                    c1 = c[k] * (x[:, 1] - y0[k]) ** 2

                    PotEn += A[k] * np.exp(a1 + b1 + c1)
            elif x.ndim == 1:
                PotEn = 0
                for k in range(4):
                    a1 = a[k] * (x[0] - x0[k]) ** 2
                    b1 = b[k] * (x[0] - x0[k]) * (x[1] - y0[k])
                    c1 = c[k] * (x[1] - y0[k]) ** 2

                    PotEn += A[k] * np.exp(a1 + b1 + c1)
        elif isinstance(x, torch.Tensor):
            PotEn = torch.zeros_like(x[:, 0], dtype=torch.float32)
            for k in range(4):
                a1 = a[k] * (x[:, 0] - x0[k]) ** 2
                b1 = b[k] * (x[:, 0] - x0[k]) * (x[:, 1] - y0[k])
                c1 = c[k] * (x[:, 1] - y0[k]) ** 2

                PotEn += A[k] * torch.exp(a1 + b1 + c1)

        return PotEn

    def gradient(self, x):
        """
        Calculate the gradient of the Müller potential for a given input array x.

        Parameters:
        x (ndarray): Input array of shape (n, 3) where n is the number of particles.

        Returns:
        ndarray: The gradient of the Müller potential evaluated at x.
        """
        # Calculate potential value
        x0 = np.array([1, 0, -0.5, -1])
        y0 = np.array([0, 0.5, 1.5, 1])

        a = np.array([-1, -1, -6.5, 0.7])
        b = np.array([0, 0, 11, 0.6])
        c = np.array([-10, -10, -6.5, 0.7])
        A = np.array([-200, -100, -170, 15])

        # Compute gradients using finite differences
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                grad = np.zeros_like(x)
                for k in range(4):

                    a1 = a[k] * (x[:, 0] - x0[k]) ** 2
                    b1 = b[k] * (x[:, 0] - x0[k]) * (x[:, 1] - y0[k])
                    c1 = c[k] * (x[:, 1] - y0[k]) ** 2
                    Var1 = 2 * a[k] * (x[:, 0] - x0[k]) + \
                        b[k] * (x[:, 1] - y0[k])
                    Var2 = 2 * c[k] * (x[:, 1] - y0[k]) + \
                        b[k] * (x[:, 0] - x0[k])

                    grad[:, 0] += A[k] * np.exp(a1 + b1 + c1) * Var1
                    grad[:, 1] += A[k] * np.exp(a1 + b1 + c1) * Var2
                return grad
            elif x.ndim == 1:
                grad = np.zeros_like(x)
                for k in range(4):

                    a1 = a[k] * (x[0] - x0[k]) ** 2
                    b1 = b[k] * (x[0] - x0[k]) * (x[1] - y0[k])
                    c1 = c[k] * (x[1] - y0[k]) ** 2
                    Var1 = 2 * a[k] * (x[0] - x0[k]) + b[k] * (x[1] - y0[k])
                    Var2 = 2 * c[k] * (x[1] - y0[k]) + b[k] * (x[0] - x0[k])

                    grad[0] += A[k] * np.exp(a1 + b1 + c1) * Var1
                    grad[1] += A[k] * np.exp(a1 + b1 + c1) * Var2
        elif isinstance(x, torch.Tensor):
            grad = torch.zeros_like(x, dtype=torch.float32)
            for k in range(4):
                a1 = a[k] * (x[:, 0] - x0[k]) ** 2
                b1 = b[k] * (x[:, 0] - x0[k]) * (x[:, 1] - y0[k])
                c1 = c[k] * (x[:, 1] - y0[k]) ** 2
                Var1 = 2 * a[k] * (x[:, 0] - x0[k]) + \
                    b[k] * (x[:, 1] - y0[k])
                Var2 = 2 * c[k] * (x[:, 1] - y0[k]) + \
                    b[k] * (x[:, 0] - x0[k])

                grad[:, 0] += A[k] * torch.exp(a1 + b1 + c1) * Var1
                grad[:, 1] += A[k] * torch.exp(a1 + b1 + c1) * Var2

        return grad

    def c_a(self):
        return np.array([-0.5582, 1.4417])

    def c_b(self):
        return np.array([0.6235, 0.0281])

    def in_a(self, x, r):
        center = np.array([[-0.5582, 1.4417]])
        if x.ndim == 2:
            return np.sum((x - center)**2, axis=1) < r**2
        elif x.ndim == 1:
            return np.sum((x - center)**2) < r**2

    def in_b(self, x, r):
        center = np.array([[0.6235, 0.0281]])
        if x.ndim == 2:
            return np.sum((x - center)**2, axis=1) < r**2
        elif x.ndim == 1:
            return np.sum((x - center)**2) < r**2

    def points_on_a_boundary(self, num, r):
        theta = 2 * np.pi / num * np.arange(num)
        center = np.array([[-0.5582, 1.4417]])
        return center + r * np.array([np.cos(theta), np.sin(theta)]).T

    def points_in_a(self, num, r):
        theta = 2 * np.pi / num * np.arange(num)
        center = np.array([[-0.5582, 1.4417]])
        rr = r * np.random.uniform(size=(num, 1))
        return center + np.array([np.cos(theta), np.sin(theta)]).T * rr

    def points_on_b_boundary(self, num, r):
        theta = 2 * np.pi / num * np.arange(num)
        center = np.array([[0.6235, 0.0281]])
        return center + np.array([np.cos(theta), np.sin(theta)]).T * r

    def points_in_b(self, num, r):
        theta = 2 * np.pi / num * np.arange(num)
        center = np.array([[0.6235, 0.0281]])
        rr = r * np.random.uniform(size=(num, 1))
        return center + np.array([np.cos(theta), np.sin(theta)]).T * rr


@numba.njit()
def Muller_grad(x):
    """
    Calculate the gradient of the Müller potential for a given input array x.

    Parameters:
    x (ndarray): Input array of shape (n, 3) where n is the number of particles.

    Returns:
    ndarray: The gradient of the Müller potential evaluated at x.
    """
    # Calculate potential value

    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])

    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    A = np.array([-200, -100, -170, 15])

    # Compute gradients using finite differences

    if x.ndim == 2:
        grad = np.zeros_like(x)
        for k in range(4):

            a1 = a[k] * (x[:, 0] - x0[k]) ** 2
            b1 = b[k] * (x[:, 0] - x0[k]) * (x[:, 1] - y0[k])
            c1 = c[k] * (x[:, 1] - y0[k]) ** 2
            Var1 = 2 * a[k] * (x[:, 0] - x0[k]) + \
                b[k] * (x[:, 1] - y0[k])
            Var2 = 2 * c[k] * (x[:, 1] - y0[k]) + \
                b[k] * (x[:, 0] - x0[k])

            grad[:, 0] += A[k] * np.exp(a1 + b1 + c1) * Var1
            grad[:, 1] += A[k] * np.exp(a1 + b1 + c1) * Var2
        return grad
    elif x.ndim == 1:
        grad = np.zeros_like(x)
        for k in range(4):

            a1 = a[k] * (x[0] - x0[k]) ** 2
            b1 = b[k] * (x[0] - x0[k]) * (x[1] - y0[k])
            c1 = c[k] * (x[1] - y0[k]) ** 2
            Var1 = 2 * a[k] * (x[0] - x0[k]) + b[k] * (x[1] - y0[k])
            Var2 = 2 * c[k] * (x[1] - y0[k]) + b[k] * (x[0] - x0[k])

            grad[0] += A[k] * np.exp(a1 + b1 + c1) * Var1
            grad[1] += A[k] * np.exp(a1 + b1 + c1) * Var2
        return grad


def print_exp():
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])

    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    A = np.array([-200, -100, -170, 15])
    for i in range(4):

        print(f'{a[i]:+.1f}*(x{-x0[i]:+.1f})*(x{-x0[i]:+.1f})', end='')
        print(f'{b[i]:+.1f}*(x{-x0[i]:+.1f})*(y{-y0[i]:+.1f})', end='')
        print(f'{c[i]:+.1f}*(y{-y0[i]:+.1f})*(y{-y0[i]:+.1f})',)


# Example usage
if __name__ == "__main__":
    # Create an instance of the Müller potential
    print_exp()
    muller = MullerPotential()

    # Define input array (e.g., coordinates of particles)
    n = 100
    x = np.linspace(-1.5, 1.2, n)  # 100 points from -5 to 5
    y = np.linspace(-0.2, 2, n)  # 100 points from -5 to 5
    X, Y = np.meshgrid(x, y)

    XX = np.reshape(X, -1)
    YY = np.reshape(Y, -1)
    XXX = np.reshape(XX, X.shape)
    U = muller.potential(torch.from_numpy(np.array([XX, YY]).T))
    U = U.numpy()
    UU = np.reshape(U, X.shape)
    UU[UU > 0] = 0
    dU = muller.gradient(torch.from_numpy(np.array([XX, YY]).T))
    dU = dU.numpy()
    dx = np.reshape(dU[:, 0], X.shape)
    dy = np.reshape(dU[:, 1], X.shape)
    print(dU)
    # Calculate potential

    # np.save('muller.npy', np.array([XX, YY, U, dU[:, 0], dU[:, 1]]).T)

    abpoints = muller.points_in_a(num=100)

    U = muller.potential(abpoints)
    dU = muller.gradient(abpoints)

    plt.scatter(abpoints[:, 0], abpoints[:, 1])
    plt.show()

    # Draw potential

    plt.figure(figsize=(8, 6))
    contour = plt.contour(
        X,
        Y,
        UU, levels=20,
        cmap='turbo')  # 20 contour levels
    plt.colorbar(contour)  # Add a colorbar to indicate the scale
    plt.title('2D Contour Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.quiver(X, Y, dx, dy, color='r')
    plt.scatter(abpoints[:, 0], abpoints[:, 1], color='r')
    plt.grid(True)
    plt.show()
