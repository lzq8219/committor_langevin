import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from muller_potential import MullerPotential
import scipy.sparse

N = 1000
N_short = N - 1

xmax = 1.2
xmin = -1.5
ymax = 2
ymin = -.2

hx = (xmax - xmin) / N
hy = (ymax - ymin) / N

kbt = 10
x = np.linspace(xmin, xmax, N + 1)
y = np.linspace(ymin, ymax, N + 1)
TWP = MullerPotential()

x_short = x[1:-1]
y_short = y[1:-1]
print(x_short)
X, Y = np.meshgrid(x_short, y_short)

points = np.array([X.reshape(-1), Y.reshape(-1)]).T

r = 0.2
mask_a = TWP.in_a(points, r)
mask_b = TWP.in_b(points, r)

b_vec = -np.ones(N_short**2) * 0

b_vec[mask_a] = 0
b_vec[mask_b] = 1

print(np.any(mask_a), np.any(mask_b))

A = scipy.sparse.lil_matrix((N_short**2, N_short**2))
dU = TWP.gradient(points)
# dU = np.zeros_like(dU)


for i in range(N_short):
    for j in range(N_short):
        idx = i + j * N_short
        if mask_a[idx] or mask_b[idx]:
            A[idx, idx] = 1
        else:
            itself_repo = 0

            if i != N_short - 1:
                A[idx, idx + 1] = -dU[idx, 0] / (2 * hx) + kbt / hx**2
            else:
                itself_repo += -dU[idx, 0] / (2 * hx) + kbt / hx**2

            if i != 0:
                A[idx, idx - 1] = dU[idx, 0] / (2 * hx) + kbt / hx**2
            else:
                itself_repo += dU[idx, 0] / (2 * hx) + kbt / hx**2

            if j != N_short - 1:
                A[idx, idx + N_short] = -dU[idx, 1] / (2 * hy) + kbt / hy**2
            else:
                itself_repo += -dU[idx, 1] / (2 * hy) + kbt / hy**2

            if j != 0:
                A[idx, idx - N_short] = dU[idx, 1] / (2 * hy) + kbt / hy**2
            else:
                itself_repo += dU[idx, 1] / (2 * hy) + kbt / hy**2

            A[idx, idx] = kbt * (-2 / hx**2 - 2 / hy**2) + itself_repo

v = scipy.sparse.linalg.spsolve(A.tocsr(), b_vec)
V = v.reshape(X.shape)

'''
dxV = (V[:, 1:] - V[:, 0:-1]) / hx
dxV = dxV[:-1, :]
dyV = (V[1:, :] - V[0:-1, :]) / hy
dyV = dyV[:, :-1]
x1 = X[:-1, :-1]
y1 = Y[:-1, :-1]
u1 = np.reshape(TWP.potential(
    np.array([x1.reshape(-1), y1.reshape(-1)]).T), x1.shape)
w1 = np.exp(-u1 / kbt)
w1 = w1 / np.sum(w1)
Iu = np.sum(w1 * (dxV**2 / 2 * kbt + dyV**2 / 2 * kbt - V[:-1, :-1]))
print(f"optimal functional value: {Iu}")
'''


out = np.concatenate((points, np.expand_dims(v, axis=0).T), axis=1)
filename = f'.\\model\\fd_kbt{kbt}.txt'
np.savetxt(filename, out)

# plt.streamplot(x1, y1, dxV, dyV)
plt.contour(X, Y, V, levels=20)

plt.colorbar()
XX = X.reshape(-1)
YY = Y.reshape(-1)
u = np.reshape(TWP.potential(np.array([XX, YY]).T), X.shape)
u[u > 0] = 0

# Draw potential


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
plt.show()
