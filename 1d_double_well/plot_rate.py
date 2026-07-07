import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman','Times','DejaVu Serif']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

width_in = 5
height_in = 4



df = pd.read_csv('1d_double_well/errors_rates_1.txt')
gamma_inv = [1.0 / gamma for gamma in df['gamma']]
rates = df['NN rate']
rates_pinn = df['PINN rate']

q0 = np.loadtxt('1d_double_well/model/fd_kbt0.1_q0.txt')
kbt = 0.1
dx = 0.0005
xmax = 1
xmin = -1
Nx = int((xmax - xmin) / dx)
plt.figure(figsize=(width_in, height_in))

x = np.linspace(xmin, xmax, Nx + 1)
x = x[1:-1]
U = (x[1:-1]**2 - 1)**2
p = np.exp(-U / kbt)
p = p / np.sum(p)
dq0dx = q0[2:] - q0[:-2]
dq0dx = dq0dx / (2 * dx)
rate_q0 = np.sum((p * dq0dx**2))*kbt
rate_from_q0 = [rate_q0/gamma for gamma in df['gamma']] 
plt.plot(gamma_inv, rates, marker='o', label='NN rate')
plt.plot(gamma_inv, rate_from_q0, marker='s', label='Asymptotic estimation from q0')
plt.plot(gamma_inv, rates_pinn, marker='^', label='PINN rate')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\gamma^{-1}$')
plt.ylabel('Transition rate')
plt.legend(loc='upper right')
plt.savefig('1d_double_well/figures/rates_1.pdf', bbox_inches='tight')
plt.savefig('1d_double_well/figures/rates_1.png',dpi=300, bbox_inches='tight')