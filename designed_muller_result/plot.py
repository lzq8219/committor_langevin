
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
nrows, ncols = 2, 3
subplot_w, subplot_h = 5, 4
fig = plt.figure(figsize=(ncols*subplot_w, nrows*subplot_h))
axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]

labels = list('abcdefghijklmnopqrstuvwxyz')
for ax, lab in zip(axes, labels):
    ax.text(-0.1, 1.08, f'({lab})', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, fontweight='bold')



gammas = [5,25]
kbts = [5,5]

idx_row = -1
for gamma,kbt in zip(gammas,kbts):
    idx_row +=1
    if os.path.exists(f'designed_muller_result/figure/gamma{gamma}_kbt{kbt}') is False:
        os.makedirs(f'designed_muller_result/figure/gamma{gamma}_kbt{kbt}')
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

    model_file_ref = f'./designed_muller_result/model/gamma{gamma}_kbt{kbt}_ref.pth'
    config_file_ref = f'./designed_muller_result/config/gamma{gamma}_kbt{kbt}_ref.txt'
    q_ref = load_model(model_file_ref,config_file_ref)

    #q0 = np.loadtxt(f'./muller_potential/model/fd_kbt{kbt}.txt')

    # In[ ]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    potential = MullerPotential()
    q.to(device)
    q_ref.to(device)

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
    points = points[uu<=30,:]
    '''
    c = np.arange(len(points))
    plt.scatter(points[:, 0], points[:, 1], c=c)
    plt.colorbar()
    plt.show()
    '''
    N_matrix = Nrow * Ncol

    


    vs = np.loadtxt(f'./designed_muller_result/model/simulation_{kbt}/simulation_vconfig_kbt{kbt}.txt')
    v_sample = vs.shape[0]


    # In[ ]:



    # Generate random data for each subplot  
    
    mm=1
    nn=1  
    q.to(device)
    qqq_NNs = []
    qqq_refs = []
    qqq_pinns = []
    e_NNs = []
    e_pinns = []
    lq_NNs = []
    lq_refs = []
    elq_NNs = []
    lq_pinns = []

    d_points = np.zeros(shape=(points.shape[0],2*ndim),dtype=np.float32)
    d_points[:,:ndim] = points
    d_points = torch.from_numpy(d_points).to(device)
    


    for i in range(mm):  
        for j in range(nn):  
            idx = j+nn*i
            print(f'Calculating {idx}')
            vvm1 = vs[idx,:]
            d_points.requires_grad_(False)
            d_points[:,ndim:] = torch.from_numpy(vvm1).to(device)
            #print(d_points)
            d_points.requires_grad_(True)
            dU1 = potential.gradient(d_points[:,:ndim])
            qqq_NN = q(d_points)
            pinn_NN = pinn_loss(qqq_NN,d_points,dU1,args)
            pinn_NN = pinn_NN.detach().squeeze().cpu().numpy()
            qqq_NN = qqq_NN.detach().squeeze().cpu().numpy()

            qqq_ref = q_ref(d_points)
            pinn_ref = pinn_loss(qqq_ref,d_points,dU1,args)
            pinn_ref = pinn_ref.detach().squeeze().cpu().numpy()
            qqq_ref = qqq_ref.detach().squeeze().cpu().numpy()


            qqq_NNs.append(qqq_NN)
            qqq_refs.append(qqq_ref)
            e_NNs.append(np.abs(qqq_NN - qqq_ref))
            lq_NNs.append(pinn_NN)
            lq_refs.append(pinn_ref)
            elq_NNs.append(np.abs(pinn_NN - pinn_ref))
            

            
            

    print('Finished calculation!')
    print('Drawing figures...')
    save_fig_name_NN = f'designed_muller_result/figure/gamma{gamma}_kbt{kbt}/NN.png'
    save_fig_name_NN_ref = f'designed_muller_result/figure/gamma{gamma}_kbt{kbt}/NN_ref.png'

    
    sc=axes[idx_row*ncols+ 0].scatter(points[:,0],points[:,1],c=qqq_NNs[0])  
    axes[idx_row*ncols+ 0].contour(
        X,
        Y,
        UU, levels=10,colors = 'white') 
    #axes[idx_row, 0].set_title(f'Scatter Plot {idx+1}')  
    axes[idx_row*ncols+ 0].set_xlabel('x1')  
    axes[idx_row*ncols+ 0].set_ylabel('x2')
    fig.colorbar(sc, ax=axes[idx_row*ncols+ 0])
    sc=axes[idx_row*ncols+ 1].scatter(points[:,0],points[:,1],c=qqq_refs[0])  
    axes[idx_row*ncols+ 1].contour(
        X,
        Y,
        UU, levels=10,colors = 'white') 
    #axes[idx_row, 0].set_title(f'Scatter Plot {idx+1}')  
    axes[idx_row*ncols+ 1].set_xlabel('x1')  
    axes[idx_row*ncols+1].set_ylabel('x2')
    fig.colorbar(sc, ax=axes[idx_row*ncols+1])

    sc=axes[idx_row*ncols+ 2].scatter(points[:,0],points[:,1],c=e_NNs[0])  
    axes[idx_row*ncols+ 2].contour(
        X,
        Y,
        UU, levels=10,colors = 'white') 
    #axes[idx_row, 0].set_title(f'Scatter Plot {idx+1}')  
    axes[idx_row*ncols+ 2].set_xlabel('x1')  
    axes[idx_row*ncols+2].set_ylabel('x2')
    fig.colorbar(sc, ax=axes[idx_row*ncols+2])



    #draw_slice(mm,nn,points,qqq_NNs,X,Y,UU,save_fig_name_NN)
    #draw_slice(mm,nn,points,qqq_refs,X,Y,UU,save_fig_name_NN_ref)
   
plt.tight_layout()
plt.savefig('designed_muller_result/figure/plot_1.pdf', bbox_inches='tight')   # vector output
plt.savefig('designed_muller_result/figure/plot_1.png', bbox_inches='tight')   # alternate

    

    