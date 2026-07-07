import numpy as np
from scipy import integrate


def integrate_on_unit_triangle(func, n=5):
    def step(x): return np.array([integrate.fixed_quad(
        lambda y: func(xt, y), 0, 1 - xt, n=n)[0] for xt in x])
    return integrate.fixed_quad(step, 0, 1, n=n)[0]


def integrate_triangle(func, triangle, n=5):
    p0, p1, p2 = triangle[0], triangle[1], triangle[2]
    S = np.abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
               (p2[0] - p0[0]) * (p1[1] - p0[1]))

    def _func_on_unit_triangle(x, y):
        xx = p0[0] + (p1[0] - p0[0]) * x + (p2[0] - p0[0]) * y
        yy = p0[1] + (p1[1] - p0[1]) * x + (p2[1] - p0[1]) * y
        return func(xx, yy)
    return integrate_on_unit_triangle(_func_on_unit_triangle, n=n) * S


def dilichlet_function_unit_triangle(x, y):
    return 1 - x - y


def basis_function(triangle, i):
    p0, p1, p2 = triangle[i, :], triangle[(
        i + 1) % 3, :], triangle[(i + 2) % 3, :]
    d1, d2 = p1 - p0, p2 - p0
    S = d1[0] * d2[1] - d1[1] * d2[0]
    return lambda x, y: (1 - (d2[1] * (x - p0[0]) - d2[0] * (y - p0[1])) /
                         S - (-d1[1] * (x - p0[0]) + d1[0] * (y - p0[1])) / S)


def dx_basis_function(triangle, i):
    p0, p1, p2 = triangle[i, :], triangle[(
        i + 1) % 3, :], triangle[(i + 2) % 3, :]
    d1, d2 = p1 - p0, p2 - p0
    S = d1[0] * d2[1] - d1[1] * d2[0]
    return lambda x, y: (d1[1] - d2[1]) / S


def dy_basis_function(triangle, i):
    p0, p1, p2 = triangle[i, :], triangle[(
        i + 1) % 3, :], triangle[(i + 2) % 3, :]
    d1, d2 = p1 - p0, p2 - p0
    S = d1[0] * d2[1] - d1[1] * d2[0]
    return lambda x, y: (-d1[0] + d2[0]) / S


if __name__ == '__main__':
    triangle = np.array([[0, 0], [0, 1], [1, 0]])

    t = dx_basis_function(triangle, 1)
    print(t(0, 0), t(1, 1), t(3, 4))
