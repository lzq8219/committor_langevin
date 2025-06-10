import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import scipy.sparse


# V(x) = (x_1^2 -1)^2 + 0.3 \sum_{i=2}^d x_i^2
# A : x<-1 , B : x > 1

xmin, xmax = -1, 1
vmin, vmax = -5, 5
dx = 0.001
dv = 0.01
Nx = int((xmax - xmin) / dx)
Nv = int((vmax - vmin) / dv)
kbt = 1
gamma = 1

Ncol = Nx - 1
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


b_vec = np.zeros(N_matrix)

# dilichlet boundary
b_vec[(Ncol - 1)::Ncol] = -1 / dx * points[(Ncol - 1)::Ncol, 1] / 2

'''
c = np.arange(len(points))
plt.scatter(points[:, 0], points[:, 1], c=c[:])
plt.colorbar()
plt.show()
'''
epsilon = .001

b_vec[(Ncol - 1)::Ncol] -= epsilon * gamma * kbt / dx**2


for i in range(Nrow):
    for j in range(Ncol):
        idx = j + i * Ncol

        itself_repo = 0

        if j != Ncol - 1:
            A[idx, idx + 1] = points[idx, 1] / dx / \
                2 + epsilon * gamma * kbt / dx**2

        if j != 0:
            A[idx, idx - 1] = -points[idx, 1] / \
                dx / 2 + epsilon * gamma * kbt / dx**2

        if i != Nrow - 1:
            A[idx, idx + Ncol] = (-dU[idx] - gamma *
                                  points[idx, 1]) / (2 * dv) + gamma * kbt / dv**2
        else:
            itself_repo += (-dU[idx] - gamma * points[idx, 1]
                            ) / (2 * dv) + gamma * kbt / dv**2

        if i != 0:
            A[idx, idx - Ncol] = (+dU[idx] + gamma *
                                  points[idx, 1]) / (2 * dv) + gamma * kbt / dv**2
        else:
            itself_repo += (+dU[idx] + gamma * points[idx, 1]
                            ) / (2 * dv) + gamma * kbt / dv**2

        A[idx, idx] = -gamma * kbt * 2 / dv**2 - \
            epsilon * gamma * kbt * 2 / dx**2 + itself_repo

q = scipy.sparse.linalg.spsolve(A.tocsr(), b_vec)
Q = q.reshape(X.shape)

plt.scatter(points[:, 0], points[:, 1], c=q)
plt.colorbar()
plt.show()

plt.plot(xcal, Q[int(0 / 2), :])
plt.show()
