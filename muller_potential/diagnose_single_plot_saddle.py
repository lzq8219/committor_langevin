
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
nrows, ncols = 1,2
subplot_w, subplot_h = 5, 4
fig = plt.figure(figsize=(ncols*subplot_w, nrows*subplot_h))
axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]

labels = list('abcdefghijklmnopqrstuvwxyz')
for ax, lab in zip(axes, labels):
    ax.text(-0.1, 1.08, f'({lab})', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, fontweight='bold')


gammas = [5,25]
kbt = 5
idx = -1
for gamma in gammas:
    idx+=1
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



    model_file = f'./designed_muller_result/model/gamma{gamma}_kbt{kbt}.pth'
    config_file = f'./designed_muller_result/config/gamma{gamma}_kbt{kbt}.txt'
    q = load_model(model_file,config_file)

    model_file_pinn = f'./muller_potential/model/gamma{gamma}_kbt{kbt}_pinn.pth'
    config_file_pinn = f'./muller_potential/config/gamma{gamma}_kbt{kbt}_pinn.txt'
    q_pinn = load_model(model_file_pinn,config_file_pinn)

    q0 = np.loadtxt(f'./muller_potential/model/fd_kbt{kbt}.txt')
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
    q_vgrid_grid = q_vgrid.reshape(v_x.shape)
    q_pinn_vgrid = q_pinn(d_vgrid).detach().squeeze().to('cpu').numpy()


    sc = axes[idx].scatter(vs_grid[:,0],vs_grid[:,1],c=q_vgrid)
    axes[idx].contour(
                v_x,
                v_y,
                q_vgrid_grid, levels=[0.4,0.6], colors='orange', linewidths=3)

    fig.colorbar(sc, ax=axes[idx])
    axes[idx].set_xlabel('$v_1$')
    axes[idx].set_ylabel('$v_2$')
    axes[idx].set_xlim([vmin,vmax])
    axes[idx].set_ylim([vmin,vmax])

    '''
    sc = axes[idx+3].scatter(vs_grid[:,0],vs_grid[:,1],c=q_pinn_vgrid)
    fig.colorbar(sc, ax=axes[idx+3])
    axes[idx+3].set_xlabel('$v_1$')
    axes[idx+3].set_ylabel('$v_2$')
    axes[idx+3].set_xlim([vmin,vmax])
    axes[idx+3].set_ylim([vmin,vmax])
    '''






        # In[ ]:



        # Generate random data for each subplot  
        


        



        #draw_slice(mm,nn,points,qqq_NNs,X,Y,UU,save_fig_name_NN)
        #draw_slice(mm,nn,points,qqq_refs,X,Y,UU,save_fig_name_NN_ref)
   
plt.tight_layout()
plt.savefig('muller_potential/figure/plot_saddle.pdf', bbox_inches='tight')   # vector output
plt.savefig('muller_potential/figure/plot_saddle.png', bbox_inches='tight',dpi=300)   # alternate
plt.show()

    

    