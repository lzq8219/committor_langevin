import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import scipy.sparse
import numba

xmin, xmax = -1, 1
vmin, vmax = -5, 5
dx = 0.001
dv = 0.01
Nx = int((xmax - xmin) / dx)
Nv = int((vmax - vmin) / dv)
kbt = 1
gamma = 100

Ncol = Nx + 1
Nrow = Nv + 1
x = np.linspace(xmin, xmax, Nx + 1)
v = np.linspace(vmin, vmax, Nv + 1)

if Ncol == Nx - 1:
    xcal = x[1:-1]
else:
    xcal = x

if Nrow == Nv - 1:
    vcal = v[1:-1]
else:
    vcal = v


X, V = np.meshgrid(xcal, vcal)
print(X.shape, V)

points = np.array([X.reshape(-1), V.reshape(-1)]).T
'''
c = np.arange(len(points))
plt.scatter(points[:, 0], points[:, 1], c=c)
plt.colorbar()
plt.show()
'''
N_matrix = Nrow * Ncol
A = scipy.sparse.lil_matrix((N_matrix, N_matrix))

dU = 4 * (points[:, 0]**2 - 1) * points[:, 0]
print(dU.shape)

mask_a = (points[:, 0] == xmin)
mask_b = (points[:, 0] == xmax)

b_vec = np.zeros(N_matrix)

# dilichlet boundary
b_vec[(Ncol - 1)::Ncol] = -1 / dx * points[(Ncol - 1)::Ncol, 1] / 2
'''
c = np.arange(len(points))
plt.scatter(points[:, 0], points[:, 1], c=c[:])
plt.colorbar()
plt.show()
'''
epsilon = 1e-3
b_vec[(Ncol - 1)::Ncol] -= epsilon * gamma * kbt / dx**2


@numba.njit
def make_matrix(Nrow, Ncol, points, dU, b_vec, gamma, kbt, epsilon):
    data = []
    row = []
    col = []
    for i in range(Nrow):
        for j in range(Ncol):
            idx = j + i * Ncol

            itself_repo = 0

            if j != Ncol - 1:
                row.append(idx)
                col.append(idx + 1)
                data.append(points[idx, 1] / dx /
                            2 + epsilon * gamma * kbt / dx**2)

            if j != 0:
                row.append(idx)
                col.append(idx - 1)
                data.append(-points[idx, 1] /
                            dx / 2 + epsilon * gamma * kbt / dx**2)

            if i != Nrow - 1:
                row.append(idx)
                col.append(idx + Ncol)
                data.append((-dU[idx] - gamma *
                             points[idx, 1]) / (2 * dv) + gamma * kbt / dv**2)

            else:
                itself_repo += (-dU[idx] - gamma * points[idx, 1]
                                ) / (2 * dv) + gamma * kbt / dv**2

            if i != 0:
                row.append(idx)
                col.append(idx - Ncol)
                data.append((+dU[idx] + gamma *
                             points[idx, 1]) / (2 * dv) + gamma * kbt / dv**2)
            else:
                itself_repo += (+dU[idx] + gamma * points[idx, 1]
                                ) / (2 * dv) + gamma * kbt / dv**2

            row.append(idx)
            col.append(idx)
            data.append(-gamma * kbt * 2 / dv**2 -
                        epsilon * gamma * kbt * 2 / dx**2 + itself_repo)
    return (data, (row, col))


args = make_matrix(Nrow, Ncol, points, dU, b_vec, gamma, kbt, epsilon)
A = scipy.sparse.csr_array(args, shape=(N_matrix, N_matrix))
