
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
                UU, levels=20) 
            axs[i, j].set_title(f'Scatter Plot {idx+1}')  
            axs[i, j].set_xlabel('x1')  
            axs[i, j].set_ylabel('x2')
            fig.colorbar(sc, ax=axs[i,j])

    plt.savefig(save_fig_name,dpi = 300, bbox_inches='tight')
    plt.close()

# Configure logging
gammas = [1,5,25]
kbts = [5,5,5]
l2_losses = []
l1_losses = []
rl2_losses = []
rl1_losses = []  
linf_losses = []
l2_lqs = []
l1_lqs = []
linf_lqs = []
l2_elqs = []
l1_elqs = []
linf_elqs = []

l2_losses_pinn = []
l1_losses_pinn = []
rl2_losses_pinn = []
rl1_losses_pinn = []
linf_losses_pinn = []

l2_lqs_pinn = []
l1_lqs_pinn = []
linf_lqs_pinn = []

rates_NN =[]
rates_pinn = []

for gamma,kbt in zip(gammas,kbts):
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


    mm=5
    nn=5
    fig, axs = plt.subplots(mm, nn, figsize=(mm*7, nn*5))  
    q.to(device)

    # Generate random data for each subplot  
    
    mm=5
    nn=5  
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
    draw_slice(mm,nn,points,qqq_NNs,X,Y,UU,save_fig_name_NN)
    draw_slice(mm,nn,points,qqq_refs,X,Y,UU,save_fig_name_NN_ref)
    print('Finished drawing NN and ref figures!')

    save_fig_name_NN_error = f'designed_muller_result/figure/gamma{gamma}_kbt{kbt}/error.png'
    #save_fig_name_NN_pinn_error = f'designed_muller_result/figure/gamma{gamma}_kbt{kbt}/error_pinn.png'
    draw_slice(mm,nn,points,e_NNs,X,Y,UU,save_fig_name_NN_error)
    #draw_slice(mm,nn,points,e_pinns,X,Y,UU,save_fig_name_NN_pinn_error)
    print('Finished drawing NN and PINN error figures!')

    save_fig_name_lq = f'designed_muller_result/figure/gamma{gamma}_kbt{kbt}/lq.png'
    save_fig_name_elq = f'designed_muller_result/figure/gamma{gamma}_kbt{kbt}/elq.png'
    draw_slice(mm,nn,points,lq_NNs,X,Y,UU,save_fig_name_lq)
    draw_slice(mm,nn,points,elq_NNs,X,Y,UU,save_fig_name_elq )
    print('Finished drawing NN and PINN lq figures!')


    N_mean = 25
    v_mean = torch.randn(size=(N_mean, ndim), device=device)*np.sqrt(kbt)
    l2_loss = 0
    l2_norm = 0
    l1_loss = 0
    linf_loss = 0
    l1_norm = 0
    linf_norm = 1

    l2_lq = 0
    l1_lq = 0
    linf_lq = 0

    l2_loss_pinn = 0
    l1_loss_pinn = 0
    linf_loss_pinn = 0

    l2_lq_pinn = 0
    l1_lq_pinn = 0
    linf_lq_pinn = 0
    l2_elq = 0
    l1_elq = 0
    linf_elq = 0

    MP = MullerPotential()
    U = MP.potential(points)
    p = np.exp(-(U-min(U))/kbt)
    p = p/np.sum(p)

    for idx in range(N_mean):
        print(f'Calculating {idx}/{N_mean}', end='\r')
        vvm1 = v_mean[idx,:]
        d_points.requires_grad_(False)
        d_points[:,ndim:] = vvm1
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

        e_NN = np.abs(qqq_NN - qqq_ref)
        lq_NN = pinn_NN
        elq_NN = np.abs(pinn_NN - pinn_ref)

        l2_loss += np.sum(e_NN**2*p)
        l2_norm += np.sum(qqq_ref**2*p)
        l1_loss += np.sum(e_NN*p)
        l1_norm += np.sum(np.abs(qqq_ref)*p)
        linf_loss = np.max((np.max(e_NN),linf_loss))
        l2_lq += np.sum(lq_NN**2*p)
        l1_lq += np.sum(np.abs(lq_NN)*p)
        linf_lq = np.max((np.max(np.abs(lq_NN)),linf_lq))
        l2_elq += np.sum(elq_NN**2*p)
        l1_elq += np.sum(np.abs(elq_NN)*p)
        linf_elq = np.max((np.max(np.abs(elq_NN)),linf_lq))
    print(f'Calculating {N_mean}/{N_mean}')

    l2_loss /= N_mean
    l2_norm /= N_mean 
    l1_loss /= N_mean
    l1_norm /= N_mean    
    l1_losses.append(l1_loss)  
    rl1_losses.append(l1_loss/l1_norm)
    linf_losses.append(linf_loss)
    logging.info(f'Absolute error: {l2_loss**0.5}')
    l2_losses.append(l2_loss**0.5)
    logging.info(f'Relative error: {l2_loss**0.5/l2_norm**0.5}')
    rl2_losses.append(l2_loss**0.5/l2_norm**0.5)
    logging.info(f'l2 norm: {l2_norm**0.5}')

    l2_lq /= N_mean
    l1_lq /= N_mean
    l2_lqs.append(l2_lq**0.5)
    l1_lqs.append(l1_lq)
    linf_lqs.append(linf_lq)
    logging.info(f'l2 loss of lq: {l2_lq**0.5}')
    logging.info(f'l1 loss of lq: {l1_lq}')
    logging.info(f'linf loss of lq: {linf_lq}')


    l2_elq /= N_mean
    l1_elq /= N_mean
    l2_elqs.append(l2_elq**0.5)
    l1_elqs.append(l1_elq)
    linf_elqs.append(linf_elq)
    logging.info(f'l2 loss of elq: {l2_elq**0.5}')
    logging.info(f'l1 loss of elq: {l1_elq}')
    logging.info(f'linf loss of elq: {linf_elq}')

    
    

df = pd.DataFrame({
    'gamma': gammas,
    'kbt': kbts,
    'l2_loss': l2_losses,
    'l1_loss': l1_losses,
    'linf_loss': linf_losses,
    'rl2_loss': rl2_losses,
    'rl1_loss': rl1_losses,
    'l2_lq': l2_lqs,
    'l1_lq': l1_lqs,
    'linf_lq': linf_lqs,
    'l2_elq': l2_elqs,
    'l1_elq': l1_elqs,
    'linf_elq': linf_elqs
})
df.to_csv('designed_muller_result/loss_results.csv', index=False)
