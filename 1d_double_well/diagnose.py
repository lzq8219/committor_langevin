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

gammas = [100,25,5,1,0.2]
kbts = [0.1,0.1,0.1,0.1,0.1]
draw_plot = False
l2_losses = []
l1_losses = []
linf_losses = []
l2_loss_pinns = []
l1_loss_pinns = []
linf_loss_pinns = []

l2_losses_1 = []
l1_losses_1 = []
linf_losses_1 = []
l2_loss_pinns_1 = []
l1_loss_pinns_1 = []
linf_loss_pinns_1 = []

l2_losses_2 = []
l1_losses_2 = []
linf_losses_2 = []
l2_loss_pinns_2 = []
l1_loss_pinns_2 = []
linf_loss_pinns_2 = []

rates = []
rates_ref = []
rates_1 = []
rates_2 = []

qs =[]
qs_adaptive_false = []
qs_pinn = []

for kbt, gamma in zip(kbts, gammas):
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

    # %%
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
    plt.plot(xcal[:], Q[int((vslice - vmin) / dv), :], 'r',
            label='Numerical result with adaptive strategy')
    plt.plot(xcal[:], q0, 'purple', label='Overdamped langevin committor')
    # plt.plot(xcal[:], qmean,label = 'Average over velocity')
    plt.plot(xcal[:], Qfd[int((vslice - vmin) / dv), :],
            'b', label='Reference solution')
    plt.plot(xcal[:], Q_pinn[int((vslice - vmin) / dv), :],
            'g', label='PINN solution')
    # plt.plot(xcal[:], Q_adaptive_false[int((vslice - vmin) / dv), :],'orange', label='Solution with uniform distribution')
    # plt.plot(xcal, Q[int((vslice-vmin)/dv), :])
    plt.xlabel('x')
    plt.ylabel('q')
    plt.legend(loc='upper left', fontsize=8)
    plt.title(f'slice at v={vslice}')
    plt.savefig(f'1d_double_well/figures/kbt{kbt}_gamma{gamma}/committor_vslice{vslice}.png')
    plt.clf()


    # %%
    NN = 1
    vvms = torch.randn(size=(NN, ndim - 1), dtype=torch.float32,
                    device=device).to(device) * np.sqrt(kbt)
    xxms = torch.randn(size=(NN, ndim - 1), dtype=torch.float32,
                    device=device).to(device) * sigma * np.sqrt(kbt)

    # %%
    mm = 5
    nn = 5
    
    # fig1, axs1 = plt.subplots(mm, nn, figsize=(mm*7, nn*5))
    q.to(device)
    d_each_slice = torch.zeros(
        size=(
            fd.shape[0],
            2 * ndim),
        device=device,
        dtype=torch.float32)
    d_each_slice[:, [0, ndim]] = torch.from_numpy(fd[:, 0:2].astype(np.float32)).to(device)

    

    q_NNs = []
    e_NNs = []
    q_PINNs = []
    e_PINNs = []
    q_adaptive_falses = []
    e_adaptive_falses = []
    # Generate random data for each subplot
    if draw_plot:
        for i in range(mm):
            for j in range(nn):
                idx = j + nn * i

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
                qqq_PINN = q_pinn(d_each_slice).squeeze().to('cpu').detach().numpy()
                qqq_PINN = qqq_PINN.reshape(X.shape)
                qqq_adaptive_false = q_adaptive_false(d_each_slice).squeeze().to('cpu').detach().numpy()
                qqq_adaptive_false = qqq_adaptive_false.reshape(X.shape)
                q_NNs.append(qqq_NN)
                e_NNs.append(qqq_NN - fd[:, 2].reshape(X.shape))
                q_PINNs.append(qqq_PINN)
                e_PINNs.append(qqq_PINN - fd[:, 2].reshape(X.shape))
                q_adaptive_falses.append(qqq_adaptive_false)
                e_adaptive_falses.append(qqq_adaptive_false - fd[:, 2].reshape(X.shape))

        save_fig_name_NN = f'1d_double_well/figures/kbt{kbt}_gamma{gamma}/committor_slices.png'
        draw_slice(mm,nn,X,V,q_NNs,save_fig_name_NN,title_prefix='committor at slice')
        save_fig_name_PINN = f'1d_double_well/figures/kbt{kbt}_gamma{gamma}/committor_pinn_slices.png'
        draw_slice(mm,nn,X,V,q_PINNs,save_fig_name_PINN,title_prefix='PINN committor at slice')
        save_fig_name_adaptive_false = f'1d_double_well/figures/kbt{kbt}_gamma{gamma}/committor_adaptive_false_slices.png'
        draw_slice(mm,nn,X,V,q_adaptive_falses,save_fig_name_adaptive_false,title_prefix='committor with uniform distribution at slice')

        save_fig_name_res_NN = f'1d_double_well/figures/kbt{kbt}_gamma{gamma}/res_slices.png'
        draw_slice(mm,nn,X,V,e_NNs,save_fig_name_res_NN,title_prefix='residue at slice')
        save_fig_name_res_PINN = f'1d_double_well/figures/kbt{kbt}_gamma{gamma}/res_pinn_slices.png'
        draw_slice(mm,nn,X,V,e_PINNs,save_fig_name_res_PINN,title_prefix='PINN residue at slice')
        save_fig_name_res_adaptive_false = f'1d_double_well/figures/kbt{kbt}_gamma{gamma}/res_adaptive_false_slices.png'
        draw_slice(mm,nn,X,V,e_adaptive_falses,save_fig_name_res_adaptive_false,title_prefix='residue with uniform distribution at slice')
        

    l2_loss = 0
    l1_loss = 0
    linf_loss = 0
    l2_loss_pinn = 0
    l1_loss_pinn = 0
    linf_loss_pinn = 0
    l2_loss_adaptive_false = 0
    l1_loss_adaptive_false = 0
    linf_loss_adaptive_false = 0
    for idx in range(NN):
        print(f'{gamma} Processing 0, slice {idx+1}/{NN}')
        d_each_slice[:, (ndim + 1):] = vvms[idx].repeat(fd.shape[0], 1)
        d_each_slice[:, 1:ndim] = xxms[idx].repeat(fd.shape[0], 1)
        qqq_NN = q(d_each_slice).squeeze().to('cpu').detach().numpy()
        qqq_pinn = q_pinn(d_each_slice).squeeze().to('cpu').detach().numpy()
        qqq_adaptive_false = q_adaptive_false(d_each_slice).squeeze().to('cpu').detach().numpy()
        
        def U_func(x): return (x[:, 0]**2 - 1)**2 

        HHH = U_func(d_each_slice[:, :ndim]) + 0.5 * (d_each_slice[:, ndim]**2)
        HHH = HHH.detach().cpu().numpy()
        ppp = np.exp(-(HHH - np.min(HHH)) / kbt)
        ppp = ppp / np.sum(ppp)
        print(ppp.shape,qqq_NN.shape,fd[:,2].shape)
        l2_loss += np.sum((qqq_NN - fd[:, 2])**2 * ppp)
        l2_loss_pinn += np.sum((qqq_pinn - fd[:, 2])**2 * ppp)
        l2_loss_adaptive_false += np.sum((qqq_adaptive_false - fd[:, 2])**2 * ppp)
        l1_loss += np.sum(np.abs(qqq_NN - fd[:, 2]) * ppp)
        l1_loss_pinn += np.sum(np.abs(qqq_pinn - fd[:, 2]) * ppp)
        l1_loss_adaptive_false += np.sum(np.abs(qqq_adaptive_false - fd[:, 2]) * ppp)
        linf_loss = np.max((np.max(np.abs(qqq_NN - fd[:, 2])), linf_loss))
        linf_loss_pinn = np.max((np.max(np.abs(qqq_pinn - fd[:, 2])), linf_loss_pinn))
        linf_loss_adaptive_false = np.max((np.max(np.abs(qqq_adaptive_false - fd[:, 2])), linf_loss_adaptive_false))

    l2_loss = l2_loss / NN
    l2_loss = np.sqrt(l2_loss)
    l2_loss_pinn = l2_loss_pinn / NN
    l2_loss_pinn = np.sqrt(l2_loss_pinn)
    l2_loss_adaptive_false = l2_loss_adaptive_false / NN
    l2_loss_adaptive_false = np.sqrt(l2_loss_adaptive_false)
    l2_losses.append(l2_loss)
    l2_losses_1.append(l2_loss_pinn)
    l2_losses_2.append(l2_loss_adaptive_false)
    l1_loss = l1_loss / NN
    l1_loss_pinn = l1_loss_pinn / NN
    l1_loss_adaptive_false = l1_loss_adaptive_false / NN
    l1_losses.append(l1_loss)
    l1_losses_1.append(l1_loss_pinn)
    l1_losses_2.append(l1_loss_adaptive_false)
    linf_losses.append(linf_loss)
    linf_losses_1.append(linf_loss_pinn)
    linf_losses_2.append(linf_loss_adaptive_false)



    # %%
    
    # fig1, axs1 = plt.subplots(mm, nn, figsize=(mm*7, nn*5))

    # Generate random data for each subplot
    pinn_NNs = []
    pinn_PINNs = []
    pinn_adaptive_falses = []
    if draw_plot:
        for i in range(mm):
            for j in range(nn):
                idx = j + nn * i
                d_each_slice.requires_grad_(False)
                d_each_slice[:, (ndim + 1):] = vvms[idx].repeat(fd.shape[0], 1)
                d_each_slice[:, 1:ndim] = xxms[idx].repeat(fd.shape[0], 1)
                dU1 = dU_func(d_each_slice[:, :ndim])
                d_each_slice.requires_grad_(True)
                pinn_NN = pinn_loss(
                    q(d_each_slice),
                    d_each_slice,
                    dU1,
                    args).squeeze().detach().cpu().numpy()
                pinn_NN = pinn_NN.reshape(X.shape)
                pinn_NNs.append(pinn_NN)
                pinn_PINN = pinn_loss(
                    q_pinn(d_each_slice),
                    d_each_slice,
                    dU1,
                    args).squeeze().detach().cpu().numpy()
                pinn_PINN = pinn_PINN.reshape(X.shape)
                pinn_PINNs.append(pinn_PINN)
                pinn_adaptive_false = pinn_loss(
                    q_adaptive_false(d_each_slice),
                    d_each_slice,
                    dU1,
                    args).squeeze().detach().cpu().numpy()
                pinn_adaptive_false = pinn_adaptive_false.reshape(X.shape)
                pinn_adaptive_falses.append(pinn_adaptive_false)

        save_fig_name_PINN_NN = f'1d_double_well/figures/kbt{kbt}_gamma{gamma}/pinn_slices.png'
        draw_slice(mm,nn,X,V,pinn_NNs,save_fig_name_PINN_NN,title_prefix='pinn_loss at slice')
        save_fig_name_PINN_PINN = f'1d_double_well/figures/kbt{kbt}_gamma{gamma}/pinn_pinn_slices.png'
        draw_slice(mm,nn,X,V,pinn_PINNs,save_fig_name_PINN_PINN,title_prefix='pinn_loss of PINN at slice')
        save_fig_name_PINN_adaptive_false = f'1d_double_well/figures/kbt{kbt}_gamma{gamma}/pinn_adaptive_false_slices.png'
        draw_slice(mm,nn,X,V,pinn_adaptive_falses,save_fig_name_PINN_adaptive_false,title_prefix='pinn_loss with uniform distribution at slice')
        
    
    pinn_l2_loss = 0
    pinn_l1_loss = 0
    pinn_linf_loss = 0
    pinn_l2_loss_pinns = 0
    pinn_l1_loss_pinns = 0
    pinn_linf_loss_pinns = 0
    pinn_l2_loss_adaptive_false = 0
    pinn_l1_loss_adaptive_false = 0
    pinn_linf_loss_adaptive_false = 0
    for idx in range(NN):
        print(f'{gamma} Processing 1, slice {idx+1}/{NN}')
        d_each_slice.requires_grad_(False)
        d_each_slice[:, (ndim + 1):] = vvms[idx].repeat(fd.shape[0], 1)
        d_each_slice[:, 1:ndim] = xxms[idx].repeat(fd.shape[0], 1)
        dU1 = dU_func(d_each_slice[:, :ndim])
        d_each_slice.requires_grad_(True)
        pinn_NN = pinn_loss(
            q(d_each_slice),
            d_each_slice,
            dU1,
            args).squeeze().detach().cpu().numpy()
        pinn_pinn = pinn_loss(
            q_pinn(d_each_slice),
            d_each_slice,
            dU1,
            args).squeeze().detach().cpu().numpy()
        pinn_adaptive_false = pinn_loss(
            q_adaptive_false(d_each_slice),
            d_each_slice,
            dU1,
            args).squeeze().detach().cpu().numpy()
        def U_func(x): return (x[:, 0]**2 - 1)**2 

        HHH = U_func(d_each_slice[:, :ndim]) + 0.5 * (d_each_slice[:, ndim]**2)
        HHH = HHH.detach().cpu().numpy()
        ppp = np.exp(-(HHH - np.min(HHH)) / kbt)
        ppp = ppp / np.sum(ppp)
        pinn_l2_loss += np.sum((pinn_NN)**2 * ppp)
        pinn_l2_loss_pinns += np.sum((pinn_pinn)**2 * ppp)
        pinn_l2_loss_adaptive_false += np.sum((pinn_adaptive_false)**2 * ppp)
        pinn_l1_loss += np.sum(np.abs(pinn_NN) * ppp)
        pinn_l1_loss_pinns += np.sum(np.abs(pinn_pinn) * ppp)
        pinn_l1_loss_adaptive_false += np.sum(np.abs(pinn_adaptive_false)* ppp)
        pinn_linf_loss = np.max((np.max(np.abs(pinn_NN)), linf_loss))
        pinn_linf_loss_pinns = np.max((np.max(np.abs(pinn_pinn)), linf_loss_pinn))
        pinn_linf_loss_adaptive_false = np.max((np.max(np.abs(pinn_adaptive_false)), linf_loss_adaptive_false))

    pinn_l2_loss = pinn_l2_loss / NN
    pinn_l2_loss = np.sqrt(pinn_l2_loss)
    pinn_l1_loss = pinn_l1_loss / NN
    pinn_l2_loss_pinns = pinn_l2_loss_pinns / NN
    pinn_l2_loss_pinns = np.sqrt(pinn_l2_loss_pinns)
    pinn_l1_loss_pinns = pinn_l1_loss_pinns / NN
    pinn_l2_loss_adaptive_false = pinn_l2_loss_adaptive_false / NN
    pinn_l2_loss_adaptive_false = np.sqrt(pinn_l2_loss_adaptive_false)
    pinn_l1_loss_adaptive_false = pinn_l1_loss_adaptive_false / NN
    l2_loss_pinns.append(pinn_l2_loss)
    l1_loss_pinns.append(pinn_l1_loss)
    linf_loss_pinns.append(pinn_linf_loss)
    l2_loss_pinns_1.append(pinn_l2_loss_pinns)
    l1_loss_pinns_1.append(pinn_l1_loss_pinns)
    linf_loss_pinns_1.append(pinn_linf_loss_pinns)
    l2_loss_pinns_2.append(pinn_l2_loss_adaptive_false)
    l1_loss_pinns_2.append(pinn_l1_loss_adaptive_false)
    linf_loss_pinns_2.append(pinn_linf_loss_adaptive_false)




    def rate(model, data, weight, args, device, xdim, vdim):
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

    def rate_fd(Qfd, X,V, weight, args):
        kbt = args['kbt']
        gamma = args['gamma']
        dx = X[0, 1] - X[0, 0]
        dv = V[1, 0] - V[0, 0]

        dQdv = np.zeros_like(Qfd)


        dQdv[1:-1, :] = (Qfd[2:, :] - Qfd[:-2, :]) / (2 * dv)
        dQdv[0, :] = (Qfd[1, :] - Qfd[0, :]) / dv
        dQdv[-1, :] = (Qfd[-1, :] - Qfd[-2, :]) / dv

        integrand = dQdv**2

        rate_value = gamma * kbt * np.sum(weight * integrand)

        return rate_value

        
    def U_func(x): return (x[:, 0]**2 - 1)**2 


    w_invariant_valid = U_func(valid_data[:, :ndim])
    w_invariant_valid = w_invariant_valid - torch.min(w_invariant_valid)
    w_invariant_valid = torch.exp(-w_invariant_valid / kbt)
    w_invariant_valid = w_invariant_valid / torch.sum(w_invariant_valid)
    w_fd = (fd[:,0]**2-1)**2 + 0.5*fd[:,1]**2
    w_fd = w_fd - np.min(w_fd)
    w_fd = np.exp(-w_fd / kbt)
    w_fd = w_fd / np.sum(w_fd)
    w_fd = w_fd.reshape(X.shape)

    rate_nn = rate(q, valid_data, w_invariant_valid, args, device, ndim, ndim)
    rate_pinn = rate(
        q_pinn,
        valid_data,
        w_invariant_valid,
        args,
        device,
        ndim,
        ndim)
    rate_adaptive_false = rate(
        q_adaptive_false,
        valid_data,
        w_invariant_valid,
        args,
        device,
        ndim,
        ndim)
    rate_ref = rate_fd(
        fd[:, 2].reshape(X.shape),
        X,
        V,
        w_fd,
        args)   


    rates.append(rate_nn.item())
    rates_ref.append(rate_ref)
    rates_1.append(rate_pinn.item())
    rates_2.append(rate_adaptive_false.item())
    print(f'NN rate: {rate_nn.item()}')
    print(f'Reference rate: {rate_ref}')
    print(f'PINN rate: {rate_pinn.item()}')
    print(f'Adaptive false rate: {rate_adaptive_false.item()}')
    plt.close('all')



gammas_inv = [1.0 / gamma for gamma in gammas]
plt.plot(gammas_inv, rates, marker='o', label='NN rate')
plt.xlabel('1/\\gamma')
plt.ylabel('Transition rate')
plt.legend(loc='upper right')
plt.savefig('1d_double_well/figures/rates_vs_gamma_inv.png')
df = pd.DataFrame({
    'kbt': kbts,
    'gamma': gammas,
    'Ref rate': rates_ref,
    'NN rate': rates,
    'PINN rate': rates_1,
    'Adaptive false rate': rates_2,
    'l2 loss': l2_losses,
    'l1 loss': l1_losses,
    'linf loss': linf_losses,
    'l2 pinn loss': l2_loss_pinns,
    'l1 pinn loss': l1_loss_pinns,
    'linf pinn loss': linf_loss_pinns,
    'l2 loss pinn': l2_losses_1,
    'l1 loss pinn': l1_losses_1,
    'linf loss pinn': linf_losses_1,
    'l2 pinn loss pinn': l2_loss_pinns_1,
    'l1 pinn loss pinn': l1_loss_pinns_1,
    'linf pinn loss pinn': linf_losses_1,
    'l2 loss adaptive false': l2_losses_2,
    'l1 loss adaptive false': l1_losses_2,
    'linf loss adaptive false': linf_losses_2,
    'l2 pinn loss adaptive false': l2_loss_pinns_2,
    'l1 pinn loss adaptive false': l1_loss_pinns_2,
    'linf pinn loss adaptive false': linf_loss_pinns_2
})
df.to_csv('1d_double_well/errors_rates_1.txt', index=False)