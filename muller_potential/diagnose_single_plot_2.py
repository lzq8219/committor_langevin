
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



# Configure logging
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman','Times','DejaVu Serif']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Figure size in inches (example for 2x6 grid)
nrows, ncols = 3,3
subplot_w, subplot_h = 8, 4





def draw_slice(nrows,ncols,points,c,X,Y,UU,save_fig_name,subplot_w=5, subplot_h=4):
    fig = plt.figure(figsize=(ncols*subplot_w, nrows*subplot_h))
    axs = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
    labels = list('abcdefghijklmnopqrstuvwxyz')
    for ax, lab in zip(axs, labels):
        ax.text(-0.1, 1.08, f'({lab})', transform=ax.transAxes,
                ha='left', va='top', fontsize=10, fontweight='bold')
    for i in range(nrows):  
        for j in range(ncols):  
            idx = j+ncols*i
            
            # Create scatter plot  
            sc=axs[idx].scatter(points[:,0],points[:,1],c=c[idx],cmap='viridis',marker='s')  
            contour_lines = axs[idx].contour(
                X,
                Y,
                UU, levels=10, colors = 'white') 
             
            plt.clabel(contour_lines, inline=True, fontsize=8)
            #axs[idx].set_title(f'Scatter Plot {idx+1}')  
            axs[idx].set_xlabel('x1')  
            axs[idx].set_ylabel('x2')
            fig.colorbar(sc, ax=axs[idx])

    plt.savefig(save_fig_name,dpi = 300, bbox_inches='tight')
    plt.close()

# Configure logging
gammas = [1,5,25]
kbts = [5,5,5]
calculating_avarege = False

qqq_NNs=[]

points = 0
for gamma,kbt in zip(gammas,kbts):
    if os.path.exists(f'muller_potential/figure/gamma{gamma}_kbt{kbt}') is False:
        os.makedirs(f'muller_potential/figure/gamma{gamma}_kbt{kbt}')
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



    model_file = f'./muller_potential/model/gamma{gamma}_kbt{kbt}.pth'
    config_file = f'./muller_potential/config/gamma{gamma}_kbt{kbt}.txt'
    q = load_model(model_file,config_file)

    model_file_pinn = f'./muller_potential/model/gamma{gamma}_kbt{kbt}_pinn.pth'
    config_file_pinn = f'./muller_potential/config/gamma{gamma}_kbt{kbt}_pinn.txt'
    q_pinn = load_model(model_file_pinn,config_file_pinn)

    q0 = np.loadtxt(f'./muller_potential/model/fd_kbt{kbt}.txt')

    # In[ ]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    potential = MullerPotential()
    q.to(device)
    q_pinn.to(device)

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
    # print(X.shape, V)

    points = np.array([X.reshape(-1), Y.reshape(-1)]).T.astype(np.float32)
    UU = potential.potential(points).reshape(X.shape)
    UU[UU>0] = 0
    '''
    c = np.arange(len(points))
    plt.scatter(points[:, 0], points[:, 1], c=c)
    plt.colorbar()
    plt.show()
    '''
    N_matrix = Nrow * Ncol
  
    vs = np.loadtxt(f'./muller_potential/model/simulation_{kbt}/simulation_vconfig_kbt{kbt}.txt')
    v_sample = vs.shape[0]
    q.to(device)

    simulation = np.loadtxt(f'./muller_potential/model/simulation_{kbt}/simulation_kbt{kbt}_gamma{gamma}_0_2.txt')
    points = simulation[:,:ndim]
    U = potential.potential(points)
    mask_simulation = U<=10
    points = points[mask_simulation,:]
    d_points = np.zeros(shape=(points.shape[0],2*ndim),dtype=np.float32)

    d_points[:,:ndim] = points
    d_points = torch.from_numpy(d_points).to(device)
    idx = 0

    vvm1 = vs[idx,:]
    
    simulation = simulation[mask_simulation,:]
    d_points.requires_grad_(False)
    d_points[:,ndim:] = torch.from_numpy(vvm1).to(device)
    #print(d_points)
    qqq_NN = q(d_points)
    qqq_pinn = q_pinn(d_points)

    qqq_NN = qqq_NN.detach().squeeze().cpu().numpy()
    qqq_pinn = qqq_pinn.detach().squeeze().cpu().numpy()
    qqq_NNs.append(qqq_NN)
    qqq_NNs.append(qqq_pinn)
    qqq_NNs.append(simulation[:,ndim])

save_fig_name = f'muller_potential/figure/compare_qqq_gammas.png'
draw_slice(nrows,ncols,points,qqq_NNs,X,Y,UU,save_fig_name,subplot_w=5, subplot_h=4)