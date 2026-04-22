
import sys
import os
origin_directory = os.getcwd()
model_directory = os.path.join(origin_directory, 'muller_potential')
src_directory = os.path.join(origin_directory, 'src')
sys.path.append(src_directory)
sys.path.append(model_directory)
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from nn import FunctionModel, save_model, load_model
import copy
import matplotlib.pyplot as plt
from muller_potential import MullerPotential
from model_training import train_resample,pinn_loss,build_rightside
from hist import hist_reweight
import pandas as pd
import logging
import matplotlib as mpl

def draw_slice(mm,nn,points,c,X,Y,UU,save_fig_name):
    fig, axs = plt.subplots(mm, nn, figsize=(mm*7, nn*5))
    for i in range(mm):  
        for j in range(nn):  
            idx = j+nn*i
            
            # Create scatter plot  
            sc=axs[i, j].scatter(points[:,0],points[:,1],c=c[idx])  
            axs[i, j].contour(
                X,
                Y,
                UU, levels=10) 
            axs[i, j].set_title(f'Scatter Plot {idx+1}')  
            axs[i, j].set_xlabel('x1')  
            axs[i, j].set_ylabel('x2')
            fig.colorbar(sc, ax=axs[i,j])

    plt.savefig(save_fig_name,dpi = 300, bbox_inches='tight')
    plt.close()

# Configure logging
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman','Times','DejaVu Serif']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Figure size in inches (example for 2x6 grid)
nrows, ncols = 1,3
subplot_w, subplot_h = 5, 4
fig = plt.figure(figsize=(ncols*subplot_w, nrows*subplot_h))
axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]

labels = list('abcdefghijklmnopqrstuvwxyz')
for ax, lab in zip(axes, labels):
    ax.text(-0.1, 1.08, f'({lab})', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, fontweight='bold')


gamma = 5
kbt = 5



ndim = 2

lam = 10
eta = 10
omega = gamma

args = {
        "ndim": ndim,
        "gamma": gamma,
        "kbt": kbt,
        "lam": lam,
        "eta": eta,
        "omega": omega
    }

committor_analytical = np.loadtxt(
            f'muller_potential/model/simulation_{kbt}/simulation_kbt{kbt}_gamma{gamma}_mask_2.txt')
test_points = committor_analytical[:,0:2]

model_file = f'./designed_muller_result/model/gamma{gamma}_kbt{kbt}.pth'
config_file = f'./designed_muller_result/config/gamma{gamma}_kbt{kbt}.txt'
q = load_model(model_file,config_file)

model_file_pinn = f'./muller_potential/model/gamma{gamma}_kbt{kbt}_pinn.pth'
config_file_pinn = f'./muller_potential/config/gamma{gamma}_kbt{kbt}_pinn.txt'
q_pinn = load_model(model_file_pinn,config_file_pinn)

q0 = np.loadtxt(f'./muller_potential/model/fd_kbt{kbt}.txt')
grid = q0[:,0:2]
N_grid = int(np.sqrt(grid.shape[0]))
X_grid = grid[:,0].reshape(N_grid, N_grid)
Y_grid = grid[:,1].reshape(N_grid, N_grid)
q0_grid = q0[:,2].reshape(N_grid, N_grid)
q0 = q0[::33,:]

# In[ ]:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
potential = MullerPotential()
q.to(device)
q_pinn.to(device)

    #q0 = np.loadtxt(f'./muller_potential/model/fd_kbt{kbt}.txt')

    # In[ ]:


xmin, xmax = -1.5, 1.2
ymin, ymax = -0.2, 2
dx = 0.01
dy = 0.01
Nx = int((xmax - xmin) / dx)
Ny = int((ymax - ymin) / dy)


Ncol = Nx + 1
Nrow = Ny + 1
x = np.linspace(xmin, xmax, Nx + 1)
y = np.linspace(ymin, ymax, Ny + 1)

if Ncol == Nx - 1:
    xcal = x[1:-1]
else:
    xcal = x

if Nrow == Ny - 1:
    ycal = y[1:-1]
else:
    ycal = y


X, Y = np.meshgrid(xcal, ycal)
points = np.array([X.reshape(-1), Y.reshape(-1)]).T.astype(np.float32)

# print(X.shape, V)

points = np.array([X.reshape(-1), Y.reshape(-1)]).T.astype(np.float32)
uu = potential.potential(points)

UU = potential.potential(points).reshape(X.shape)
UU[UU>0] = 0

u = potential.potential(q0[:,0:2])
mask = u<100

saddle_point = torch.tensor([[-0.822, 0.624]],dtype=torch.float32).to(device)

vmax = 3 * np.sqrt(kbt)
vmin = -3 * np.sqrt(kbt)
dv = 0.05 * np.sqrt(kbt)
Nv = int((vmax - vmin) / dv)
v = np.linspace(vmin, vmax, Nv + 1)
v_x, v_y = np.meshgrid(v, v)
vs_grid = np.array([v_x.reshape(-1), v_y.reshape(-1)]).T.astype(np.float32)
vs_grid_torch = torch.from_numpy(vs_grid).to(device)

d_vgrid = torch.zeros(size=(vs_grid_torch.shape[0],2*ndim)).to(device)
d_vgrid[:,:ndim] = saddle_point.repeat(vs_grid_torch.shape[0],1)
d_vgrid[:,ndim:] = vs_grid_torch

q_vgrid = q(d_vgrid).detach().squeeze().to('cpu').numpy()
q_pinn_vgrid = q_pinn(d_vgrid).detach().squeeze().to('cpu').numpy()





qqq = np.loadtxt(f'muller_potential/model/ave_qqq_kbt{kbt}_gamma{gamma}.txt')
qqq_grid = qqq.reshape(N_grid, N_grid)
qqq = qqq[::33]
qqq_pinn = np.loadtxt(f'muller_potential/model/ave_qqq_pinn_kbt{kbt}_gamma{gamma}.txt')
qqq_pinn_grid = qqq_pinn.reshape(N_grid, N_grid)
qqq_pinn=qqq_pinn[::33]
lq=np.loadtxt(f'muller_potential/model/ave_lq_kbt{kbt}_gamma{gamma}.txt')
lq=lq[::33]
lq_pinn=np.loadtxt(f'muller_potential/model/ave_lq_pinn_kbt{kbt}_gamma{gamma}.txt')
lq_pinn = lq_pinn[::33]

axes[0].contour(
            X,
            Y,
            UU, levels=10, colors = 'white',zorder=2)  # 20 contour levels
sc = axes[0].scatter(q0[:, 0], q0[:, 1], c=q0[:,2],zorder=1)
axes[0].contour(X_grid, Y_grid, q0_grid, levels=[0.5], colors='orange', linewidths=3,zorder=2)

axes[0].scatter(test_points[:,0], test_points[:,1], c='red', s=10,zorder=3)
axes[0].set_xlabel('$x_1$')
axes[0].set_ylabel('$x_2$')
fig.colorbar(sc, ax=axes[0])

axes[1].contour(
            X,
            Y,
            UU, levels=10, colors = 'white',zorder=2)  # 20 contour levels
sc = axes[1].scatter(q0[:, 0], q0[:, 1], c=qqq,zorder=1)
axes[1].contour(X_grid, Y_grid, qqq_grid, levels=[0.5], colors='orange', linewidths=3,zorder=2)

axes[1].scatter(test_points[:,0], test_points[:,1], c='red', s=10,zorder=3)
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
fig.colorbar(sc, ax=axes[1])



axes[2].contour(
            X,
            Y,
            UU, levels=10, colors = 'white',zorder=2)  # 20 contour levels
sc = axes[2].scatter(q0[:, 0], q0[:, 1], c=qqq_pinn,zorder=1)
axes[2].contour(X_grid, Y_grid, qqq_pinn_grid, levels=[0.5], colors='orange', linewidths=3,zorder=2)
axes[2].scatter(test_points[:,0], test_points[:,1], c='red', s=10,zorder=3)
axes[2].set_xlabel('$x_1$')
axes[2].set_ylabel('$x_2$')
fig.colorbar(sc, ax=axes[2])









    # In[ ]:



    # Generate random data for each subplot  
    


    



    #draw_slice(mm,nn,points,qqq_NNs,X,Y,UU,save_fig_name_NN)
    #draw_slice(mm,nn,points,qqq_refs,X,Y,UU,save_fig_name_NN_ref)
   
plt.tight_layout()
plt.savefig('muller_potential/figure/plot_1.pdf', bbox_inches='tight')   # vector output
plt.savefig('muller_potential/figure/plot_1.png', bbox_inches='tight',dpi=300)   # alternate

    

    