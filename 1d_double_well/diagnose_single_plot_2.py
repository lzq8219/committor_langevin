# %%
import sys
import os


sys.path.append('src')

from model_training import train_resample, pinn_loss, build_rightside, train_pinn
import matplotlib.pyplot as plt
import copy
from nn import FunctionModel, save_model, load_model
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
import torch
import sys
import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Styling / fonts
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman','Times','DejaVu Serif']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Figure size in inches (example for 2x6 grid)
nrows, ncols = 4, 5
subplot_w, subplot_h = 5,5









def draw_slice(mm,nn,X,V,qqq_s,save_fig_name,mm_length=7,nn_length=5,x_label='x',y_label='v',title_prefix='committor at slice'):
    fig = plt.figure(figsize=(ncols*subplot_w, nrows*subplot_h))
    axs = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
    labels = list('abcdefghijklmnopqrstuvwxyz')
    for ax, lab in zip(axs, labels):
        ax.text(-0.1, 1.02, f'({lab})', transform=ax.transAxes,
                ha='left', va='top', fontsize=10, fontweight='bold')
    for i in range(mm):
        for j in range(nn):
            idx = j + nn * i
            # Create scatter plot
            qqq = qqq_s[idx]
            sc = axs[idx].contourf(X, V, qqq, levels=5)
            # sc1 = axs1[i, j].scatter(points[:,0],points[:,1],c=np.abs(qqq-fd[:,2]))
            # axs[i, j].set_title(f'{title_prefix} {idx+1}')
            axs[idx].set_xlabel(x_label)
            axs[idx].set_ylabel(y_label)
            # axs1[i, j].set_title(f'residue at slice {idx+1}')
            # axs1[i, j].set_xlabel('x')
            # axs1[i, j].set_ylabel('v')
            fig.colorbar(sc, ax=axs[idx])
            # fig1.colorbar(sc1, ax=axs1[i,j])
    plt.savefig(save_fig_name)
    plt.close()

gammas = [0.2,1,5,25]
kbts = [0.1,0.1,0.1,0.1]
ndim = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigma = 1/0.3

q_NNs = []
e_NNs = []
q_PINNs = []
e_PINNs = []
q_adaptive_falses = []
e_adaptive_falses = []


kbt = 0.1
xmin, xmax = -1, 1
vmin, vmax = -3 * np.sqrt(kbt), 3 * np.sqrt(kbt)
dx = 0.0005
dv = 0.01 * np.sqrt(kbt)
Nx = int((xmax - xmin) / dx)
Nv = int((vmax - vmin) / dv)


Ncol = Nx - 1
Nrow = Nv + 1
x = np.linspace(xmin, xmax, Nx + 1)
x = x.astype(np.float32)
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
for kbt, gamma in zip(kbts, gammas):
    NN = 5
    vvms = torch.randn(size=(NN, ndim - 1), dtype=torch.float32,
                    device=device).to(device) * np.sqrt(kbt)
    xxms = torch.randn(size=(NN, ndim - 1), dtype=torch.float32,
                    device=device).to(device) * sigma * np.sqrt(kbt)

    # %%
    model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}_pinn.pth'
    config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}_pinn.txt'
    q = load_model(model_file, config_file)
    
    # fig1, axs1 = plt.subplots(mm, nn, figsize=(mm*7, nn*5))
    q.to(device)
    fd = np.loadtxt(f'1d_double_well/model/fd_g{gamma}_kbt{kbt}.txt')
    d_each_slice = torch.zeros(
        size=(
            fd.shape[0],
            2 * ndim),
        device=device,
        dtype=torch.float32)
    d_each_slice[:, [0, ndim]] = torch.from_numpy(fd[:, 0:2].astype(np.float32)).to(device)

    

    
    # Generate random data for each subplot
    for idx in range(ncols):


        '''
        ddd.requires_grad_(False)
        ddd[:,(ndim):] = vvm1
        ddd.requires_grad_(True)
        dU1 = potential.gradient(ddd[:,:ndim])
        pinn_l = pinn_loss(q(ddd),ddd,dU1,args)
        pinn_l = pinn_l.detach().cpu().numpy()
        qqq1 = q(ddd).squeeze().to('cpu').detach()
        '''

        d_each_slice[:, (ndim + 1):] = vvms[idx].repeat(fd.shape[0], 1)
        d_each_slice[:, 1:ndim] = xxms[idx].repeat(fd.shape[0], 1)
        qqq_NN = q(d_each_slice).squeeze().to('cpu').detach().numpy()
        qqq_NN = qqq_NN.reshape(X.shape)
        
        q_NNs.append(qqq_NN)
        e_NNs.append(qqq_NN - fd[:, 2].reshape(X.shape))
        



save_fig_name_NN = f'1d_double_well/figures/committor_pinn_slices_with_gammas.png'
draw_slice(nrows,ncols,X,V,q_NNs,save_fig_name_NN,title_prefix='committor at slice', mm_length=subplot_h, nn_length=subplot_w, x_label='x', y_label='v')

    
    



