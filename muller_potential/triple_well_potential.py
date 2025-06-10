import numpy as np
import torch
import matplotlib.pyplot as plt
import numba as nb


class TripleWellPotential:
    def __init__(self):
        self.a = np.array([-1, 0])
        self.b = np.array([1, 0])
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
        x0 = np.array([0, 0, 1, -1])
        y0 = np.array([1 / 3, 5 / 3, 0, 0])

        a = np.array([3, -3, -5, -5])

        # Calculate potential energy
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                PotEn = np.zeros_like(x[:, 0])
                PotEn = x[:, 0]**4 / 5 + (x[:, 1] - 1 / 3)**4 / 5
                for k in range(4):
                    PotEn += a[k] * np.exp(-(x[:, 0] - x0[k])
                                           ** 2 - (x[:, 1] - y0[k])**2)
            elif x.ndim == 1:
                PotEn = x[0]**4 / 5 + (x[1] - 1 / 3)**4 / 5
                for k in range(4):
                    PotEn += a[k] * \
                        np.exp(-(x[0] - x0[k])**2 - (x[1] - y0[k])**2)
        elif isinstance(x, torch.Tensor):
            PotEn = torch.zeros_like(x[:, 0], dtype=torch.float32)
            PotEn = x[:, 0]**4 / 5 + (x[:, 1] - 1 / 3)**4 / 5
            for k in range(4):
                PotEn += a[k] * torch.exp(-(x[:, 0] - x0[k])
                                          ** 2 - (x[:, 1] - y0[k])**2)
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
        x0 = np.array([0, 0, 1, -1])
        y0 = np.array([1 / 3, 5 / 3, 0, 0])
        points = np.array([x0, y0]).T

        a = np.array([3, -3, -5, -5])

        # Calculate potential energy
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                grad = np.zeros_like(x)
                grad[:, 0] = (x[:, 0])**3 * 4 / 5
                grad[:, 1] = (x[:, 1] - 1 / 3)**3 * 4 / 5
                for k in range(4):
                    p = a[k] * np.exp(-(x[:, 0] - x0[k]) **
                                      2 - (x[:, 1] - y0[k])**2)
                    # print(x.shape, points[k, :].shape)
                    grad[:, 0] += -p * 2 * (x[:, 0] - x0[k])
                    grad[:, 1] += -p * 2 * (x[:, 1] - y0[k])
            elif x.ndim == 1:
                grad = (x - np.array([0, 1 / 3]))**3 * 4 / 5
                for k in range(4):
                    p = a[k] * np.exp(-(x[0] - x0[k])**2 - (x[1] - y0[k])**2)
                    # print(x.shape, points[k, :].shape)
                    grad[0] += -p * 2 * (x[0] - x0[k])
                    grad[1] += -p * 2 * (x[1] - y0[k])
        elif isinstance(x, torch.Tensor):
            grad = torch.zeros_like(x, dtype=torch.float32)
            grad[:, 0] = (x[:, 0])**3 * 4 / 5
            grad[:, 1] = (x[:, 1] - 1 / 3)**3 * 4 / 5
            for k in range(4):
                p = a[k] * torch.exp(-(x[:, 0] - x0[k]) **
                                     2 - (x[:, 1] - y0[k])**2)
                # print(x.shape, points[k, :].shape)
                grad[:, 0] += -p * 2 * (x[:, 0] - x0[k])
                grad[:, 1] += -p * 2 * (x[:, 1] - y0[k])
        return grad

    def c_a(self):
        return self.a

    def c_b(self):
        return self.b

    def in_a(self, x, r):
        center = self.a
        if x.ndim == 2:
            return np.sum((x - center)**2, axis=1) < r**2
        elif x.ndim == 1:
            return np.sum((x - center)**2) < r**2

    def in_b(self, x, r):
        center = self.b
        if x.ndim == 2:
            return np.sum((x - center)**2, axis=1) < r**2
        elif x.ndim == 1:
            return np.sum((x - center)**2) < r**2

    def points_on_a_boundary(self, num, r):
        theta = 2 * np.pi / num * np.arange(num)
        center = self.a
        return center + r * np.array([np.cos(theta), np.sin(theta)]).T

    def points_in_a(self, num, r):
        theta = 2 * np.pi / num * np.arange(num)
        center = self.a
        rr = r * np.random.uniform(size=(num, 1))
        return center + np.array([np.cos(theta), np.sin(theta)]).T * rr

    def points_on_b_boundary(self, num, r):
        theta = 2 * np.pi / num * np.arange(num)
        center = self.b
        return center + np.array([np.cos(theta), np.sin(theta)]).T * r

    def points_in_b(self, num, r):
        theta = 2 * np.pi / num * np.arange(num)
        center = self.b
        rr = r * np.random.uniform(size=(num, 1))
        return center + np.array([np.cos(theta), np.sin(theta)]).T * rr


@nb.njit()
def TWP_grad(x):
    """
    Calculate the gradient of the Müller potential for a given input array x.

    Parameters:
    x (ndarray): Input array of shape (n, 3) where n is the number of particles.

    Returns:
    ndarray: The gradient of the Müller potential evaluated at x.
    """
    # Calculate potential value
    x0 = np.array([0, 0, 1, -1])
    y0 = np.array([1 / 3, 5 / 3, 0, 0])

    a = np.array([3, -3, -5, -5])

    # Calculate potential energy

    if x.ndim == 2:
        grad = np.zeros_like(x)
        grad[:, 0] = (x[:, 0])**3 * 4 / 5
        grad[:, 1] = (x[:, 1] - 1 / 3)**3 * 4 / 5
        for k in range(4):
            p = a[k] * np.exp(-(x[:, 0] - x0[k]) **
                              2 - (x[:, 1] - y0[k])**2)
            # print(x.shape, points[k, :].shape)
            grad[:, 0] += -p * 2 * (x[:, 0] - x0[k])
            grad[:, 1] += -p * 2 * (x[:, 1] - y0[k])
    elif x.ndim == 1:
        grad = (x - np.array([0, 1 / 3]))**3 * 4 / 5
        for k in range(4):
            p = a[k] * np.exp(-(x[0] - x0[k])**2 - (x[1] - y0[k])**2)
            # print(x.shape, points[k, :].shape)
            grad[0] += -p * 2 * (x[0] - x0[k])
            grad[1] += -p * 2 * (x[1] - y0[k])
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
    # print_exp()
    TWP = TripleWellPotential()

    # Define input array (e.g., coordinates of particles)
    n = 20
    x = np.linspace(-2, 2, n)  # 100 points from -5 to 5
    y = np.linspace(-1, 2, n)  # 100 points from -5 to 5
    X, Y = np.meshgrid(x, y)

    XX = np.reshape(X, -1)
    YY = np.reshape(Y, -1)
    XXX = np.reshape(XX, X.shape)
    points = np.array([XX, YY]).T
    d = torch.from_numpy(np.array([XX, YY]).T)
    U = TWP.potential(d).numpy()
    UU = np.reshape(U, X.shape)
    dU = TWP_grad(points)
    print(dU)
    dx = np.reshape(dU[:, 0], X.shape)
    dy = np.reshape(dU[:, 1], X.shape)
    plt.scatter(XX, YY, c=U)
    plt.show()
    # Calculate potential

    # np.save('muller.npy', np.array([XX, YY, U, dU[:, 0], dU[:, 1]]).T)

    abpoints = TWP.points_in_b(num=100, r=0.2)

    # U = TWP.potential(abpoints)
    # dU = TWP.gradient(abpoints)

    # plt.scatter(abpoints[:, 0], abpoints[:, 1])
    # plt.show()

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
