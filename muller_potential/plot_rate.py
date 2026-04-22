import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from muller_potential import MullerPotential

df = pd.read_csv('muller_potential/loss_results.csv')
gamma_inv = [1.0 / gamma for gamma in df['gamma']]
rates = df['rates_NN']
rates_pinn = df['rates_pinn']
kbt = 5

q01 = np.loadtxt(f'./muller_potential/model/fd_kbt{kbt}.txt')


N = 1000
N_short = N - 1
q0 = q01[:,2].reshape(N_short, N_short)

xmax = 1.2
xmin = -1.5
ymax = 2
ymin = -.2

hx = (xmax - xmin) / N
hy = (ymax - ymin) / N

kbt = 5
x = np.linspace(xmin, xmax, N + 1)
y = np.linspace(ymin, ymax, N + 1)
muller = MullerPotential()

x_short = x[1:-1]
y_short = y[1:-1]
X, Y = np.meshgrid(x_short, y_short)

points = np.array([X.reshape(-1), Y.reshape(-1)]).T
print(np.sum((points[:,0]-q01[:,0])**2))
print(np.sum((points[:,1]-q01[:,1])**2))
U = muller.potential(points).reshape(N_short, N_short)
p = np.exp(-U / kbt)
p = p / np.sum(p)
p = p.reshape(N_short, N_short)
p = p[1:-1,1:-1]
dq0dx = q0[2:,1:-1] - q0[:-2,1:-1]
dq0dx = dq0dx / (2 * hx)
dq0dy = q0[1:-1,2:] - q0[1:-1,:-2]
dq0dy = dq0dy / (2 * hy)


rate_q0 = np.sum((p * dq0dx**2 + p * dq0dy**2)) * kbt
rate_from_q0 = [rate_q0/gamma for gamma in df['gamma']] 
plt.plot(gamma_inv, rates, marker='o', label='NN rate')
plt.plot(gamma_inv, rates_pinn, marker='^', label='PINN rate')
plt.plot(gamma_inv, rate_from_q0, marker='s', label='Asymptotic estimation from q0')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\gamma^{-1}$')
plt.ylabel('Transition rate')
plt.legend(loc='upper left')
plt.savefig('muller_potential/figure/rates.png')