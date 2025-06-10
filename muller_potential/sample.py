import numpy as np
from triple_well_potential import TripleWellPotential, TWP_grad
from muller_potential import MullerPotential, Muller_grad
import matplotlib.pyplot as plt
import numba
import time
from langevin_simulation import ul_simulation
from hist import hist_reweight

muller = MullerPotential()
grad_func = muller.gradient
kbt = 20
print(muller.c_b())
st = time.time()
xs, vs = ul_simulation(grad_func, 2,1, kbt, nstep= 10 **7
                        , stride=10, xinit=None)
xs = xs.reshape((xs.shape[0],xs.shape[2]))
end = time.time()
print(f'Using time {end-st}s!')

# plt.scatter(xs[:, 0], xs[:, 1], alpha=0.5)
np.savetxt(f'model/muller_{kbt}_A_1e7.txt', xs)
ngrid = 400
grid = np.linspace(-2, 2, ngrid)
y, x = np.meshgrid(grid, grid)
y = y.flatten()
x = x.flatten()
g = np.array([x, y]).T
# filename='simulation/long/COLVAR'
data = xs

# fes=calculateFES_multi(df,grid,16)
nstart = 0
h = hist_reweight(data, np.ones_like(data[:, 0]), -2, 2, -2, 2, ngrid)

h = h.flatten()
cc = np.log(h[h > 0])
thread = -20
cc[cc < thread] = thread
plt.scatter(g[:, 0][h > 0], g[:, 1][h > 0],
            cmap='turbo', c=cc, s=1)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='FES')
plt.show()

