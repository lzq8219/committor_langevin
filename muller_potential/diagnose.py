
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
            sc=axs[i, j].scatter(points[:,0],points[:,1],c=c[idx],cmap='viridis',marker='s')  
            contour_lines = axs[i, j].contour(
                X,
                Y,
                UU, levels=10, colors = 'white') 
             
            plt.clabel(contour_lines, inline=True, fontsize=8)
            axs[i, j].set_title(f'Scatter Plot {idx+1}')  
            axs[i, j].set_xlabel('x1')  
            axs[i, j].set_ylabel('x2')
            fig.colorbar(sc, ax=axs[i,j])

    plt.savefig(save_fig_name,dpi = 300, bbox_inches='tight')
    plt.close()

# Configure logging
gammas = [5]
kbts = [5]
calculating_avarege = False

l2_losses = []
l1_losses = []
rl2_losses = []
rl1_losses = []  
linf_losses = []
l2_lqs = []
l1_lqs = []
linf_lqs = []

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

    d_grid = torch.zeros(size=(q0.shape[0],2*ndim)).to(device)

    d_grid[:,:ndim] = torch.from_numpy(q0[:,:ndim]).to(device)
    d_grid[:,ndim:] = 0
    qqq = torch.zeros(size=(d_grid.shape[0],1),dtype=torch.float32,device=device)
    lq = torch.zeros(size=(d_grid.shape[0],1),dtype=torch.float32,device=device)
    qqq_pinn = torch.zeros(size=(d_grid.shape[0],1),dtype=torch.float32,device=device)
    lq_pinn = torch.zeros(size=(d_grid.shape[0],1),dtype=torch.float32,device=device)
    NNN = 1000

    #with torch.no_grad():
    if calculating_avarege:         
        for ttt in range(NNN):
            print(f'Calculating average committor value: {ttt+1}/{NNN}',end='\r')
            d_grid.requires_grad_(False)
            d_grid[:,ndim:] = torch.randn(size=(1,ndim),device=device)*torch.ones(size=(d_grid.shape[0],ndim),device=device)*np.sqrt(kbt)
            d_grid.requires_grad_(True)
            #print(ddd.shape)
            dU1 = potential.gradient(d_grid[:,:ndim])
            temp = q(d_grid)
            #print(ddd.shape,temp.shape)
            temp_pinn = pinn_loss(temp,d_grid,dU1,args)
            with torch.no_grad():
            
                qqq += temp
                lq += temp_pinn
            del temp,temp_pinn
            temp = q_pinn(d_grid)
            temp_pinn = pinn_loss(temp,d_grid,dU1,args)
            with torch.no_grad():
                
                qqq_pinn += temp
                lq_pinn += temp_pinn
            del temp,temp_pinn
            

        print('')
        print('Completed!')
        qqq = qqq/NNN
        lq = lq/NNN
        qqq_pinn = qqq_pinn/NNN
        lq_pinn = lq_pinn/NNN

        
        qqq = qqq.squeeze().to('cpu').detach().numpy()
        lq = lq.squeeze().to('cpu').detach().numpy()
        qqq_pinn = qqq_pinn.squeeze().to('cpu').detach().numpy()
        lq_pinn = lq_pinn.squeeze().to('cpu').detach().numpy()

        np.savetxt(f'muller_potential/model/ave_qqq_kbt{kbt}_gamma{gamma}.txt',qqq)
        np.savetxt(f'muller_potential/model/ave_qqq_pinn_kbt{kbt}_gamma{gamma}.txt',qqq_pinn)
        np.savetxt(f'muller_potential/model/ave_lq_kbt{kbt}_gamma{gamma}.txt',lq)
        np.savetxt(f'muller_potential/model/ave_lq_pinn_kbt{kbt}_gamma{gamma}.txt',lq_pinn)
    else:
        qqq = np.loadtxt(f'muller_potential/model/ave_qqq_kbt{kbt}_gamma{gamma}.txt')
        qqq_pinn = np.loadtxt(f'muller_potential/model/ave_qqq_pinn_kbt{kbt}_gamma{gamma}.txt')
        lq=np.loadtxt(f'muller_potential/model/ave_lq_kbt{kbt}_gamma{gamma}.txt')
        lq_pinn=np.loadtxt(f'muller_potential/model/ave_lq_pinn_kbt{kbt}_gamma{gamma}.txt')
    
    lq[lq>50] = 50
    lq_pinn[lq_pinn>50]=50



    ttt = np.abs(q0[:,2]-qqq)
    plt.contour(
            X,
            Y,
            UU, levels=10, colors = 'white')  # 20 contour levels
    #ttt[ttt>0.3] = 0.3
    plt.scatter(q0[:,0], q0[:,1], c=ttt)
    plt.title('$|q_0^{ref} - q_0^{NN}|$')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.savefig(f'muller_potential/figure/gamma{gamma}_kbt{kbt}/ave_error.png',dpi = 300, bbox_inches='tight')
    plt.close()
    print('Finished drawing average error figure!')

    ttt = np.abs(q0[:,2]-qqq_pinn)
    plt.contour(
            X,
            Y,
            UU, levels=10, colors = 'white')  # 20 contour levels
    #ttt[ttt>0.3] = 0.3
    plt.scatter(q0[:,0], q0[:,1], c=ttt)
    plt.title('$|q_0^{ref} - q_0^{NN}|$')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.savefig(f'muller_potential/figure/gamma{gamma}_kbt{kbt}/ave_error_pinn.png',dpi = 300, bbox_inches='tight')
    plt.close()
    print('Finished drawing average pinn error figure!')

    # In[ ]:

    fig = plt.figure(figsize=(5, 5))
    plt.contour(
            X,
            Y,
            UU, levels=10, colors = 'white')  # 20 contour levels
    plt.scatter(q0[:, 0], q0[:, 1], c=qqq)
    plt.title('Muller potential')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.savefig(f'muller_potential/figure/gamma{gamma}_kbt{kbt}/ave_muller_potential.png',dpi = 300, bbox_inches='tight')
    plt.close()
    print('Finished drawing average figure!')

    fig = plt.figure(figsize=(5, 5))
    plt.contour(
            X,
            Y,
            UU, levels=10, colors = 'white') # 20 contour levels
    plt.scatter(q0[:, 0], q0[:, 1], c=qqq_pinn)
    plt.title('Muller potential')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.savefig(f'muller_potential/figure/gamma{gamma}_kbt{kbt}/ave_muller_potential_pinn.png',dpi = 300, bbox_inches='tight')
    plt.close()
    print('Finished drawing average pinn figure!')

    fig = plt.figure(figsize=(5, 5))
    plt.contour(
            X,
            Y,
            UU, levels=10, colors = 'white')  # 20 contour levels
    plt.scatter(q0[:, 0], q0[:, 1], c=lq)
    plt.title('Muller potential')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.savefig(f'muller_potential/figure/gamma{gamma}_kbt{kbt}/ave_lq.png',dpi = 300, bbox_inches='tight')
    plt.close()
    print('Finished drawing average lq figure!')

    fig = plt.figure(figsize=(5, 5))
    plt.contour(
            X,
            Y,
            UU, levels=10, colors = 'white') # 20 contour levels
    plt.scatter(q0[:, 0], q0[:, 1], c=lq_pinn)
    plt.title('Muller potential')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.savefig(f'muller_potential/figure/gamma{gamma}_kbt{kbt}/ave_lq_pinn.png',dpi = 300, bbox_inches='tight')
    plt.close()
    print('Finished drawing average lq pinn figure!')


    vs = np.loadtxt(f'./muller_potential/model/simulation_{kbt}/simulation_vconfig_kbt{kbt}.txt')
    v_sample = vs.shape[0]


    # In[ ]:


    mm=5
    nn=5
    fig, axs = plt.subplots(mm, nn, figsize=(mm*7, nn*5))  
    q.to(device)
    simulation = np.loadtxt(f'./muller_potential/model/simulation_{kbt}/simulation_kbt{kbt}_gamma{gamma}_0_2.txt')
    
    
    points = simulation[:,:ndim]
    U = potential.potential(points)
    mask_simulation = U<=100
    points = points[mask_simulation,:]
    d_points = np.zeros(shape=(points.shape[0],2*ndim),dtype=np.float32)

    d_points[:,:ndim] = points
    d_points = torch.from_numpy(d_points).to(device)

    # Generate random data for each subplot  
    for i in range(mm):  
        for j in range(nn):  
            idx = j+nn*i
            vvm1 = vs[idx,:]
            simulation = np.loadtxt(f'./muller_potential/model/simulation_{kbt}/simulation_kbt{kbt}_gamma{gamma}_{idx}_2.txt')
            simulation = simulation[mask_simulation,:]
            # Create scatter plot  
            sc=axs[i, j].scatter(simulation[:,0],simulation[:,1],c=simulation[:,2],marker='s') 
            axs[i, j].contour(
                X,
                Y,
                UU, levels=10, colors = 'white') 
            axs[i, j].set_title(f'Scatter Plot {idx+1}')  
            axs[i, j].set_xlabel('x1')  
            axs[i, j].set_ylabel('x2')
            fig.colorbar(sc, ax=axs[i,j])

    plt.savefig(f'muller_potential/figure/gamma{gamma}_kbt{kbt}/simulation.png',dpi = 300, bbox_inches='tight')
    plt.close()



    saddle_point = torch.tensor([[-0.822, 0.624]],dtype=torch.float32).to(device)

    vmax = 3 * np.sqrt(kbt)
    vmin = -3 * np.sqrt(kbt)
    dv = 0.01 * np.sqrt(kbt)
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
    plt.scatter(vs_grid[:,0],vs_grid[:,1],c=q_vgrid)
    plt.colorbar()
    plt.title('Committor at saddle point by NN')
    plt.xlabel('v1')
    plt.ylabel('v2')
    plt.savefig(f'muller_potential/figure/gamma{gamma}_kbt{kbt}/committor_saddle_NN.png',dpi = 300, bbox_inches='tight')
    plt.close()

    plt.scatter(vs_grid[:,0],vs_grid[:,1],c=q_pinn_vgrid)
    plt.colorbar()
    plt.title('Committor at saddle point by PINN')
    plt.xlabel('v1')
    plt.ylabel('v2')
    plt.savefig(f'muller_potential/figure/gamma{gamma}_kbt{kbt}/committor_saddle_PINN.png',dpi = 300, bbox_inches='tight')
    plt.close()

    mm=5
    nn=5  
    q.to(device)
    qqq_NNs = []
    qqq_pinns = []
    e_NNs = []
    e_pinns = []
    pinn_NNs = []
    pinn_pinns =[]
    simulation = np.loadtxt(f'./muller_potential/model/simulation_{kbt}/simulation_kbt{kbt}_gamma{gamma}_0_2.txt')
    
    
    points = simulation[:,:ndim]
    U = potential.potential(points)
    mask_simulation = U<=100
    points = points[mask_simulation,:]
    d_points = np.zeros(shape=(points.shape[0],2*ndim),dtype=np.float32)

    d_points[:,:ndim] = points
    d_points = torch.from_numpy(d_points).to(device)
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


    for i in range(mm):  
        for j in range(nn):  
            idx = j+nn*i
            print(f'Calculating {idx}')
            vvm1 = vs[idx,:]
            simulation = np.loadtxt(f'./muller_potential/model/simulation_{kbt}/simulation_kbt{kbt}_gamma{gamma}_{idx}_2.txt')
            simulation = simulation[mask_simulation,:]
            d_points.requires_grad_(False)
            d_points[:,ndim:] = torch.from_numpy(vvm1).to(device)
            #print(d_points)
            d_points.requires_grad_(True)
            dU1 = potential.gradient(d_points[:,:ndim])
            qqq_NN = q(d_points)
            pinn_NN = pinn_loss(qqq_NN,d_points,dU1,args)
            pinn_NN = pinn_NN.detach().squeeze().cpu().numpy()
            qqq_NN = qqq_NN.detach().squeeze().cpu().numpy()
            qqq_NNs.append(qqq_NN)
            e_NNs.append(np.abs(qqq_NN - simulation[:,2]))
            pinn_NNs.append(pinn_NN)
            MP = MullerPotential()
            U = MP.potential(simulation[:,0:ndim])
            p = np.exp(-(U-min(U))/kbt)
            p = p/np.sum(p)
            
            print(p.shape,pinn_NNs[-1].shape,(np.abs(pinn_NNs[-1])*p).shape)

            l2_loss += np.sum(e_NNs[-1]**2*p)
            l2_norm += np.sum(simulation[:,2]**2*p)
            l1_loss += np.sum(e_NNs[-1]*p)
            l1_norm += np.sum(np.abs(simulation[:,2])*p)
            linf_loss = np.max((np.max(e_NNs[-1]),linf_loss))
            l2_lq += np.sum(pinn_NNs[-1]**2*p)
            l1_lq += np.sum(np.abs(pinn_NNs[-1])*p)
            linf_lq = np.max((np.max(np.abs(pinn_NNs[-1])),linf_lq))

            qqq_pinn = q_pinn(d_points)
            pinn_pinn = pinn_loss(qqq_pinn,d_points,dU1,args)
            pinn_pinn = pinn_pinn.detach().squeeze().cpu().numpy()
            qqq_pinn = qqq_pinn.detach().squeeze().cpu().numpy()
            qqq_pinns.append(qqq_pinn)
            e_pinns.append(np.abs(qqq_pinn - simulation[:,2]))
            pinn_pinns.append(pinn_pinn)

            l2_loss_pinn += np.sum(e_pinns[-1]**2*p)
            l1_loss_pinn += np.sum(e_pinns[-1]*p)
            linf_loss_pinn = np.max((np.max(e_pinns[-1]),linf_loss_pinn))

            l2_lq_pinn += np.sum(pinn_pinns[-1]**2*p)
            l1_lq_pinn += np.sum(np.abs(pinn_pinns[-1])*p)
            linf_lq_pinn = np.max((np.max(np.abs(pinn_pinns[-1])),linf_lq_pinn))

    print('Finished calculation!')
    print('Drawing figures...')
    save_fig_name_NN = f'muller_potential/figure/gamma{gamma}_kbt{kbt}/NN.png'
    save_fig_name_NN_pinn = f'muller_potential/figure/gamma{gamma}_kbt{kbt}/NN_pinn.png'
    draw_slice(mm,nn,points,qqq_NNs,X,Y,UU,save_fig_name_NN)
    draw_slice(mm,nn,points,qqq_pinns,X,Y,UU,save_fig_name_NN_pinn)
    print('Finished drawing NN and PINN figures!')

    save_fig_name_NN_error = f'muller_potential/figure/gamma{gamma}_kbt{kbt}/error.png'
    save_fig_name_NN_pinn_error = f'muller_potential/figure/gamma{gamma}_kbt{kbt}/error_pinn.png'
    draw_slice(mm,nn,points,e_NNs,X,Y,UU,save_fig_name_NN_error)
    draw_slice(mm,nn,points,e_pinns,X,Y,UU,save_fig_name_NN_pinn_error)
    print('Finished drawing NN and PINN error figures!')

    save_fig_name_lq = f'muller_potential/figure/gamma{gamma}_kbt{kbt}/lq.png'
    save_fig_name_lq_pinn = f'muller_potential/figure/gamma{gamma}_kbt{kbt}/lq_pinn.png'
    draw_slice(mm,nn,points,pinn_NNs,X,Y,UU,save_fig_name_lq)
    draw_slice(mm,nn,points,pinn_pinns,X,Y,UU,save_fig_name_lq_pinn)
    
    
    



    print('Finished drawing NN and PINN lq figures!')

    l2_loss /= mm*nn
    l2_norm /= mm*nn 
    l1_loss /= mm*nn
    l1_norm /= mm*nn    
    l1_losses.append(l1_loss)  
    rl1_losses.append(l1_loss/l1_norm)
    linf_losses.append(linf_loss)
    logging.info(f'Absolute error: {l2_loss**0.5}')
    l2_losses.append(l2_loss**0.5)
    logging.info(f'Relative error: {l2_loss**0.5/l2_norm**0.5}')
    rl2_losses.append(l2_loss**0.5/l2_norm**0.5)
    logging.info(f'l2 norm: {l2_norm**0.5}')

    l2_lq /= mm*nn
    l1_lq /= mm*nn
    l2_lqs.append(l2_lq**0.5)
    l1_lqs.append(l1_lq)
    linf_lqs.append(linf_lq)
    logging.info(f'l2 loss of lq: {l2_lq**0.5}')
    logging.info(f'l1 loss of lq: {l1_lq}')
    logging.info(f'linf loss of lq: {linf_lq}')

    l2_loss_pinn /= mm*nn
    l1_loss_pinn /= mm*nn
    l1_losses_pinn.append(l1_loss_pinn)
    rl1_losses_pinn.append(l1_loss_pinn/l1_norm)
    linf_losses_pinn.append(linf_loss_pinn)
    logging.info(f'Absolute error: {l2_loss_pinn**0.5}')
    l2_losses_pinn.append(l2_loss_pinn**0.5)    
    logging.info(f'Relative error: {l2_loss_pinn**0.5/l2_norm**0.5}')
    rl2_losses_pinn.append(l2_loss_pinn**0.5/l2_norm**0.5)
    logging.info(f'l2 norm: {l2_norm**0.5}')
    l2_lq_pinn /= mm*nn
    l1_lq_pinn /= mm*nn
    l2_lqs_pinn.append(l2_lq_pinn**0.5)
    l1_lqs_pinn.append(l1_lq_pinn)
    linf_lqs_pinn.append(linf_lq_pinn)
    logging.info(f'l2 loss of lq pinn: {l2_lq_pinn**0.5}')
    logging.info(f'l1 loss of lq pinn: {l1_lq_pinn}')
    logging.info(f'linf loss of lq pinn: {linf_lq_pinn}')

    
    def rate(model, data, weight, args, device):
        # data and weight should be sampled from the equilibrium distribution
        data = data.to(device)
        weight = weight.to(device)
        data.requires_grad_(True)
        kbt = args['kbt']
        gamma = args['gamma']

        qqq = model(data)
        with torch.no_grad():
            gradients = torch.autograd.grad(outputs=qqq, inputs=data,
                                            grad_outputs=torch.ones_like(qqq),
                                            create_graph=False, retain_graph=False)[0]
        if weight.shape is not (data.shape[0], 1):
            weight = weight.unsqueeze(dim=1)
        grad_v = gradients[:, ndim:]

        return gamma * kbt * torch.sum(weight * (grad_v**2))
    print('Estimating rates...')
    N_repeat = 100
    d_points = d_points.repeat(N_repeat,1)
    d_points[:,ndim:] = torch.randn(size=(d_points.shape[0],ndim),device=device)*np.sqrt(kbt)
    U = potential.potential(d_points[:,:ndim])
    p = torch.exp(-(U - torch.min(U)) / kbt)
    p = p / torch.sum(p)
    rate_NN = rate(q, d_points, p, args, device).item()
    rate_NN_pinn = rate(q_pinn, d_points, p, args, device).item()
    logging.info(f'Estimated rate by NN: {rate_NN}')
    logging.info(f'Estimated rate by PINN: {rate_NN_pinn}')
    rates_NN.append(rate_NN)
    rates_pinn.append(rate_NN_pinn)
    print('Finished estimating rates!')

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
    'rates_NN': rates_NN,
    'l2_loss_pinn': l2_losses_pinn,
    'l1_loss_pinn': l1_losses_pinn,
    'linf_loss_pinn': linf_losses_pinn,
    'rl2_loss_pinn': rl2_losses_pinn,
    'rl1_loss_pinn': rl1_losses_pinn,
    'l2_lq_pinn': l2_lqs_pinn,
    'l1_lq_pinn': l1_lqs_pinn,
    'linf_lq_pinn': linf_lqs_pinn,
    'rates_pinn': rates_pinn
})
df.to_csv('muller_potential/loss_results_1.csv', index=False)
