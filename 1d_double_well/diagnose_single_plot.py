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
nrows, ncols = 2, 3
subplot_w, subplot_h = 5, 4
fig = plt.figure(figsize=(ncols*subplot_w, nrows*subplot_h))
axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]

labels = list('abcdefghijklmnopqrstuvwxyz')
for ax, lab in zip(axes, labels):
    ax.text(-0.1, 1.02, f'({lab})', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, fontweight='bold')






def draw_slice(mm,nn,X,V,qqq_s,save_fig_name,mm_length=7,nn_length=5,x_label='x',y_label='v',title_prefix='committor at slice'):
    fig, axs = plt.subplots(mm, nn, figsize=(mm * mm_length, nn * nn_length))
    for i in range(mm):
        for j in range(nn):
            idx = j + nn * i
            # Create scatter plot
            qqq = qqq_s[idx]
            sc = axs[i, j].contourf(X, V, qqq, levels=5)
            # sc1 = axs1[i, j].scatter(points[:,0],points[:,1],c=np.abs(qqq-fd[:,2]))
            axs[i, j].set_title(f'{title_prefix} {idx+1}')
            axs[i, j].set_xlabel(x_label)
            axs[i, j].set_ylabel(y_label)
            # axs1[i, j].set_title(f'residue at slice {idx+1}')
            # axs1[i, j].set_xlabel('x')
            # axs1[i, j].set_ylabel('v')
            fig.colorbar(sc, ax=axs[i, j])
            # fig1.colorbar(sc1, ax=axs1[i,j])
    plt.savefig(save_fig_name)
    plt.close()

gammas = [0.2,5]
kbts = [0.1,0.1]

i_row = -1
for kbt, gamma in zip(kbts, gammas):
    i_row += 1
    if not os.path.exists(f'1d_double_well/figures/kbt{kbt}_gamma{gamma}'):
        os.makedirs(f'1d_double_well/figures/kbt{kbt}_gamma{gamma}')
    '''
    model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}.pth'
    config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}.txt'
    qs.append(load_model(model_file, config_file))

    model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}_pinn.pth'
    config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}_pinn.txt'
    qs_pinn.append(load_model(model_file, config_file))

    model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}_adaptive_false.pth'
    config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}_adaptive_false.txt'
    qs_adaptive_false.append(load_model(model_file, config_file))
    '''
    # %%
    ndim = 3
    lam = 10
    eta = gamma*kbt
    omega = gamma
    sigma = 1/0.3

    args = {
            "ndim": ndim,
            "gamma": gamma,
            "kbt": kbt,
            "lam": lam,
            "eta": eta,
            "omega": omega
        }


    def dU_func(x):
        dU = torch.zeros_like(x)
        dU[:, 0] = 4 * (x[:, 0]**2 - 1) * x[:, 0]
        dU[:, 1:] = x[:, 1:] / sigma**2
        return dU


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and configuration
    model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}.pth'
    config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}.txt'
    q = load_model(model_file, config_file)
    

    model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}_pinn.pth'
    config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}_pinn.txt'
    q_pinn = load_model(model_file, config_file)
    

    model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}_adaptive_false.pth'
    config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}_adaptive_false.txt'
    q_adaptive_false = load_model(model_file, config_file)
    


    q.to(device=device)
    q_pinn.to(device=device)
    q_adaptive_false.to(device=device)

    valid_sample = 10**6

    valid_xm1 = torch.randn(
        size=(
            valid_sample,
            ndim - 1),
        dtype=torch.float32) * sigma * np.sqrt(kbt)
    valid_x1 = (
        torch.rand(
            size=(
                int(valid_sample),
                1),
            dtype=torch.float32)) * 2 - 1
    valid_x = torch.concatenate((valid_x1, valid_xm1), dim=1)

    valid_v = torch.randn(
        size=(
            valid_sample,
            ndim),
        dtype=torch.float32) * np.sqrt(kbt)
    # data = torch.cat((x.repeat_interleave(Nv_sample,dim=0),v.repeat(Nx_sample,1)),dim=1)
    valid_data = torch.cat((valid_x, valid_v), dim=1)
    valid_w = torch.ones(
        size=(
            valid_data.shape[0],
            1),
        device=device,
        dtype=torch.float32)
    valid_w = valid_w / torch.sum(valid_w)

    # %%
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
    # print(X.shape, V)

    points = np.array([X.reshape(-1), V.reshape(-1)]).T.astype(np.float32)
    '''
    c = np.arange(len(points))
    plt.scatter(points[:, 0], points[:, 1], c=c)
    plt.colorbar()
    plt.show()
    '''
    N_matrix = Nrow * Ncol
    d_grid = torch.zeros((len(points), 2 * ndim), dtype=torch.float32)
    d_grid[:, [0, ndim]] = torch.from_numpy(points)

    xxm1 = torch.randn(size=(1, ndim - 1)) * np.sqrt(kbt) * sigma
    vvm1 = torch.randn(size=(1, ndim - 1)) * np.sqrt(kbt)
    d_grid[:, (ndim + 1):] = vvm1
    d_grid[:, 1:ndim] = xxm1
    d_grid = d_grid.to(device)
    


    # %%
    fd = np.loadtxt(f'1d_double_well/model/fd_g{gamma}_kbt{kbt}.txt')
    q0 = np.loadtxt(f'1d_double_well/model/fd_kbt{kbt}_q0.txt')
    q_simulation = np.loadtxt('1d_double_well/model/q_s_1d.txt')

    # %% draw qmean
    '''
    qmean = np.zeros_like(q0)
    NNN = 10000
    d_one_point = torch.zeros(size=(NNN, 2 * ndim), dtype=torch.float32)
    for i in range(qmean.shape[0]):

        d_one_point[:, ndim:] = torch.randn(
            size=(NNN, ndim), dtype=torch.float32) * np.sqrt(kbt)
        d_one_point[:, 1:ndim] = torch.randn(
            size=(NNN, ndim - 1), dtype=torch.float32) * np.sqrt(kbt) * sigma
        d_one_point[:, 0] = xcal[i] * torch.ones_like(d_one_point[:, 0])
        q_temp = q(d_one_point.to(device))
        qmean[i] = q_temp.mean().item()
    '''

    # %%
    
    qqq = q(d_grid).squeeze().to('cpu').detach()
    qqq_pinn = q_pinn(d_grid).squeeze().to('cpu').detach()
    qqq_adaptive_false = q_adaptive_false(d_grid).squeeze().to('cpu').detach()

    # %%
    vslice = 0 * vmax
    Q = qqq.reshape(X.shape)
    Q_pinn = qqq_pinn.reshape(X.shape)
    Qfd = fd[:, 2].reshape(X.shape)
    Q_adaptive_false = qqq_adaptive_false.reshape(X.shape)
    # plt.plot(xcal[::10], Q[int((vslice-vmin)/dv), ::10]-Qfd[int((vslice-vmin)/dv), ::10])
    # plt.plot(xcal[:], Q[int((vslice-vmin)/dv), :]-Qfd[int((vslice-vmin)/dv), :])
    axes[i_row*ncols+2].plot(xcal[:], Q[int((vslice - vmin) / dv), :], 'r',
            label='Ours')
    axes[i_row*ncols+2].plot(xcal[:], q0, 'purple', label='Overdamped limit')
    # plt.plot(xcal[:], qmean,label = 'Average over velocity')
    axes[i_row*ncols+2].plot(xcal[:], Qfd[int((vslice - vmin) / dv), :],
            'b', label='Reference ')
    axes[i_row*ncols+2].plot(xcal[:], Q_pinn[int((vslice - vmin) / dv), :],
            'g', label='PINNs')
    # plt.plot(xcal[:], Q_adaptive_false[int((vslice - vmin) / dv), :],'orange', label='Solution with uniform distribution')
    # plt.plot(xcal, Q[int((vslice-vmin)/dv), :])
    axes[i_row*ncols+2].set_xlabel('x')
    axes[i_row*ncols+2].set_ylabel('q')
    axes[i_row*ncols+2].legend(loc='upper left', fontsize=8)
    #axes[i_row*ncols+2].set_title(f'slice at v={vslice}')
    


    # %%
    NN = 1000
    vvms = torch.randn(size=(NN, ndim - 1), dtype=torch.float32,
                    device=device).to(device) * np.sqrt(kbt)
    xxms = torch.randn(size=(NN, ndim - 1), dtype=torch.float32,
                    device=device).to(device) * sigma * np.sqrt(kbt)

    # %%
    
    # fig1, axs1 = plt.subplots(mm, nn, figsize=(mm*7, nn*5))
    q.to(device)
    d_each_slice = torch.zeros(
        size=(
            fd.shape[0],
            2 * ndim),
        device=device,
        dtype=torch.float32)
    d_each_slice[:, [0, ndim]] = torch.from_numpy(fd[:, 0:2].astype(np.float32)).to(device)

    idx = 0


    d_each_slice[:, (ndim + 1):] = vvms[idx].repeat(fd.shape[0], 1)
    d_each_slice[:, 1:ndim] = xxms[idx].repeat(fd.shape[0], 1)
    qqq_NN = q(d_each_slice).squeeze().to('cpu').detach().numpy()
    qqq_NN = qqq_NN.reshape(X.shape)

    sc = axes[i_row*ncols].contourf(X, V, qqq_NN, levels=5)
    # sc1 = axs1[i, j].scatter(points[:,0],points[:,1],c=np.abs(qqq-fd[:,2]))
    axes[i_row*ncols].set_xlabel('x')
    axes[i_row*ncols].set_ylabel('v')
    # axs1[i, j].set_title(f'residue at slice {idx+1}')
    # axs1[i, j].set_xlabel('x')
    # axs1[i, j].set_ylabel('v')
    fig.colorbar(sc, ax=axes[i_row*ncols])

    sc = axes[i_row*ncols+1].contourf(X, V, qqq_NN - fd[:, 2].reshape(X.shape), levels=5)
    # sc1 = axs1[i, j].scatter(points[:,0],points[:,1],c=np.abs(qqq-fd[:,2]))
    axes[i_row*ncols+1].set_xlabel('x')
    axes[i_row*ncols+1].set_ylabel('v')
    # axs1[i, j].set_title(f'residue at slice {idx+1}')
    # axs1[i, j].set_xlabel('x')
    # axs1[i, j].set_ylabel('v')
    fig.colorbar(sc, ax=axes[i_row*ncols+1])
    
    





plt.tight_layout()
plt.savefig('1d_double_well/figures/plot.pdf', bbox_inches='tight')   # vector output
plt.savefig('1d_double_well/figures/plot.svg', bbox_inches='tight')   # alternate
# plt.savefig('1d_double_well/figure.png', dpi=300, bbox_inches='tight')  # raster if needed