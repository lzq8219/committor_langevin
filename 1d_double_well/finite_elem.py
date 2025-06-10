import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import scipy.sparse
from TriangleIntegral import integrate_triangle, basis_function, dx_basis_function, dy_basis_function

# V(x) = (x_1^2 -1)^2 + 0.3 \sum_{i=2}^d x_i^2
# A : x<-1 , B : x > 1

xmin, xmax = -1, 1
vmin, vmax = -10, 10
dx = .1
dv = .1
Nx = int((xmax - xmin) / dx)
Nv = int((vmax - vmin) / dv)
kbt = 1
gamma = 1
int_order = 7

Ncol = Nx + 1
Nrow = Nv + 3
x = np.linspace(xmin, xmax, Nx + 1)
v = np.linspace(vmin - dv, vmax + dv, Nv + 3)

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

boundary_mask = np.zeros_like(points[:, 0])
boundary_mask[points[:, 1] == vmin - dv] = 3
boundary_mask[points[:, 1] == vmax + dv] = 4
boundary_mask[points[:, 0] == xmin] = 1
boundary_mask[points[:, 0] == xmax] = 2

print(boundary_mask == 0)


elements = []

for i in range(Nrow - 1):
    elements += [(j, j + 1, j + Ncol)
                 for j in range(i * Ncol, (i + 1) * Ncol - 1)]
    elements += [(j, j + Ncol, j + Ncol - 1)
                 for j in range(i * Ncol + 1, (i + 1) * Ncol)]

'''
ps = []
for i in range(len(elements)):
    t = points[elements[i], :]
    mid = np.sum(t, axis=0) / 3
    ps += [t[0, :], t[1, :], t[2, :], mid]

ps = np.array(ps)
plt.scatter(ps[:, 0], ps[:, 1])
plt.show()
'''

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
def dU_func(x): return 4 * (x**2 - 1) * x


b_vec = np.zeros(N_matrix)

# dilichlet boundary


'''
c = np.arange(len(points))
plt.scatter(points[:, 0], points[:, 1], c=c[:])
plt.colorbar()
plt.show()
'''
epsilon = .0000

timed = 0
for e in elements:
    triangle = points[e, :]
    for i in range(3):
        i_idx = e[i]
        if boundary_mask[i_idx] == 0:
            wi = basis_function(triangle, i)
            wiv = dy_basis_function(triangle, i)
            wix = dx_basis_function(triangle, i)
            for j in range(3):
                k = (i + j) % 3
                k_idx = e[k]
                wkx = dx_basis_function(triangle, k)
                wkv = dy_basis_function(triangle, k)
                wk = basis_function(triangle, k)

                def f(x, v): return wi(x, v) * (v * wkx(x, v) - dU_func(x)
                                                  * wkv(x, v) - gamma * v * wkv(x, v)) - kbt * gamma * wiv(x, v) * wkv(x, v) - epsilon * kbt * gamma * wix(x, v) * wkx(x, v)

                integral = integrate_triangle(f, triangle, n=int_order)
                # print(integrate_triangle(lambda x, y: 1, triangle))
                A[i_idx, k_idx] = integral

            # def f(x, v): return v * wi(x, v) / 2
            # b_vec[i_idx] = -integrate_triangle(f, triangle, n=int_order)
            # print(integral)
        elif boundary_mask[i_idx] == 1:
            b_vec[i_idx] = 0
            A[i_idx, i_idx] = 1
        elif boundary_mask[i_idx] == 2:
            b_vec[i_idx] = 1
            A[i_idx, i_idx] = 1
        elif boundary_mask[i_idx] == 3:
            A[i_idx, i_idx] = 1
            A[i_idx, i_idx + Ncol] = -1
        elif boundary_mask[i_idx] == 4:
            A[i_idx, i_idx] = 1
            A[i_idx, i_idx - Ncol] = -1
    timed += 1
    if timed % 100 == 99:
        print(timed)

q = scipy.sparse.linalg.spsolve(A.tocsr(), b_vec)
print(points.shape, q.shape,)
Q = q.reshape(X.shape)
# q = Q.reshape(-1)

plt.scatter(points[:, 0], points[:, 1], c=q)
plt.colorbar()
plt.show()

plt.plot(xcal, Q[int(0 / 2), :])
plt.show()
