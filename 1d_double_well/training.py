# %%
import sys
import os

sys.path.append('src')
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from nn import FunctionModel, save_model, load_model
import copy
import matplotlib.pyplot as plt
from model_training import train_resample,pinn_loss,build_rightside,train_pinn


# %%
ndim = 3
gammas = [0.2,1,5,25]
kbts = [0.1,0.3]
lam = 10

for kbt in kbts:
    for gamma in gammas:

        eta = gamma*kbt
        omega = gamma
        sigma = 1/0.3

        if not os.path.exists(f"1d_double_well/fig/gamma{gamma}_kbt{kbt}"):
            os.makedirs(f"1d_double_well/fig/gamma{gamma}_kbt{kbt}")

        args = {
                "ndim": ndim,
                "gamma": gamma,
                "kbt": kbt,
                "lam": lam,
                "eta": eta,
                "omega": omega
            }


        # sample 
        Nx_sample = 1000000
        Nv_sample = 1000000
        valid_sample = 10**6
        NA = 10
        NB = 10
        Nxb_sample=100
        Nvb_sample=100

        batch_size = 2048 #not implement

        layers = [2*ndim,20,20,20,1]
        activ  = 'sigmoid'

        alpha_t = 1
        T = 200
        Nt = int(T/alpha_t)
        Nsteps = 20
        lr = 1e-3

        device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu")

        x1 = (torch.rand(size=(int(Nx_sample),1),dtype=torch.float32))*2-1
        valid_x1 = (torch.rand(size=(int(valid_sample),1),dtype=torch.float32))*2-1
        #x111 = torch.rand(size = (int(Nx_sample*0.9),1),dtype=torch.float32)*0.8-0.4
        #x1 = torch.cat((x1,x111),dim = 0)
        print(x1.shape)
        xm1 = torch.randn(size=(Nx_sample,ndim-1),dtype=torch.float32)*sigma*np.sqrt(kbt)
        valid_xm1 = torch.randn(size=(valid_sample,ndim-1),dtype=torch.float32)*sigma*np.sqrt(kbt)
        x = torch.concatenate((x1,xm1),dim=1)
        valid_x = torch.concatenate((valid_x1,valid_xm1),dim=1)
        v = torch.randn(size=(Nv_sample,ndim),dtype = torch.float32)*np.sqrt(kbt)
        valid_v = torch.randn(size=(valid_sample,ndim),dtype = torch.float32)*np.sqrt(kbt)
        #data = torch.cat((x.repeat_interleave(Nv_sample,dim=0),v.repeat(Nx_sample,1)),dim=1)
        data = torch.cat((x,v),dim=1)
        valid_data = torch.cat((valid_x,valid_v),dim=1)
        valid_w = torch.ones(size=(valid_data.shape[0],1),device = device,dtype = torch.float32)
        valid_w = valid_w/torch.sum(valid_w)
        w = torch.ones(size=(data.shape[0],1),device = device,dtype = torch.float32)
        w = w/torch.sum(w)

        w_pinn = torch.ones(size=(data.shape[0],1),device = device,dtype = torch.float32)
        w_pinn = w_pinn/torch.sum(w_pinn)
        w_adaptive_false = torch.ones(size=(data.shape[0],1),device = device,dtype = torch.float32)
        w_adaptive_false = w_adaptive_false/torch.sum(w_adaptive_false)

        def dU_func(x):
            dU = torch.zeros_like(x)
            dU[:,0] = 4*(x[:,0]**2-1)*x[:,0]
            dU[:,1:] = x[:,1:]/sigma**2
            return dU
        dU = dU_func(data[:,:ndim])
        valid_dU = dU_func(valid_data[:,:ndim])

        vb = torch.randn(size=(Nvb_sample,ndim),dtype=torch.float32)*np.sqrt(kbt)
        xA1 = torch.tensor([-1-0.1*i for i in range(NA)],dtype=torch.float32).unsqueeze(-1)
        xAm1 = torch.randn(size=(Nxb_sample,ndim-1),dtype=torch.float32)*sigma*np.sqrt(kbt)
        xA = torch.cat((xA1.repeat_interleave(Nxb_sample,dim=0),xAm1.repeat(NA,1)),dim=1)
        xA = torch.cat((xA.repeat_interleave(Nvb_sample,dim=0),vb.repeat(Nxb_sample*NA,1)),dim=1)

        xB1 = torch.tensor([1+0.1*i for i in range(NB)],dtype=torch.float32).unsqueeze(-1)
        xBm1 = torch.randn(size=(Nxb_sample,ndim-1),dtype=torch.float32)*sigma*np.sqrt(kbt)
        xB = torch.cat((xB1.repeat_interleave(Nxb_sample,dim=0),xBm1.repeat(NB,1)),dim=1)
        xB = torch.cat((xB.repeat_interleave(Nvb_sample,dim=0),vb.repeat(NB*Nxb_sample,1)),dim=1)
        labelA = 0*torch.ones_like(xA[:,0])
        labelB = 1*torch.ones_like(xB[:,0])

        data_b = torch.cat((xA,xB),dim=0)
        label_b = torch.cat((labelA,labelB),dim=0).unsqueeze(dim=1)
        print(label_b.shape)
        del xA,xB,labelA,labelB

        q = FunctionModel(layer_sizes=layers,activation=activ)
        q_pinn = FunctionModel(layer_sizes=layers,activation=activ)
        q_adaptive_false = FunctionModel(layer_sizes=layers,activation=activ)
        #model_file = f'./model/gamma10_kbt0.5_1I.pth'
        #config_file = f'./config/gamma10_kbt0.5_1I.txt'
        #q = load_model(model_file,config_file)


            

        # %%
        print(device)

        # %%
        plt.hist(x1.squeeze().numpy(),bins=20)
        plt.show()

        # %%
        torch.cuda.empty_cache()
        q_pinn.to(device)

        batch_size = 2**22
        #eta = 10
        lr = 1e-3
        # kbt = 1
        loss_list,b_loss_list,tot_loss_list,valid_pinn_loss=train_pinn(model=q_pinn,
                                                data=data,
                                                w=w_pinn,
                                                batchsize=batch_size,
                                                data_b=data_b,
                                                label_b=label_b,
                                                alpha_b=10,
                                                lr = lr,
                                                num_tsteps=600,
                                                num_epoches=Nsteps,
                                                device=device,
                                                args=args,
                                                dU=dU,
                                                xdim=ndim,
                                                vdim=ndim,
                                                checkpoint=10,
                                                adaptive=False,
                                                alpha_beta= 0.9,
                                                beta = 0.3,
                                                valid_data=valid_data,
                                                valid_w=valid_w,
                                                valid_checkpoint=1,
                                                valid_dU = valid_dU)
        # %%
        q_pinn.to(device)

        batch_size = 2**22
        #eta = 10
        lr = 1e-4
        # kbt = 1
        loss_list,b_loss_list,tot_loss_list,valid_pinn_loss=train_pinn(model=q_pinn,
                                                data=data,
                                                w=w_pinn,
                                                batchsize=batch_size,
                                                data_b=data_b,
                                                label_b=label_b,
                                                alpha_b=10,
                                                lr = lr,
                                                num_tsteps=8000,
                                                num_epoches=Nsteps,
                                                device=device,
                                                args=args,
                                                dU=dU,
                                                xdim=ndim,
                                                vdim=ndim,
                                                checkpoint=10,
                                                adaptive=False,
                                                alpha_beta= 0.9,
                                                beta = 0.3,
                                                valid_data=valid_data,
                                                valid_w=valid_w,
                                                valid_checkpoint=40,
                                                valid_dU = valid_dU)

        # %%
        fff = f"1d_double_well/model/valid_loss_gamma{gamma}_kbt{kbt}_pinn.txt"
        np.savetxt(fff, valid_pinn_loss)
        t = np.arange(len(valid_pinn_loss))*40
        plt.figure(figsize=(8,6))
        plt.plot(t,valid_pinn_loss)
        plt.yscale("log")
        plt.xlabel("epochs")
        plt.ylabel("Test loss")
        plt.savefig(f"1d_double_well/fig/gamma{gamma}_kbt{kbt}/valid_loss_pinn.png")
        plt.clf()



        # %%
        args['lam'] = .2
        args['eta'] = .2
        args['omega'] = gamma

        # %%
        ## initialize

        q.to(device)

        batch_size = 2**22
        #eta = 10
        lr = 1e-3
        # kbt = 1
        loss_list,b_loss_list,tot_loss_list,pinn_loss_list,valid_pinn_loss=train_resample(model=q,
                                                data=data,
                                                w=w,
                                                batchsize=batch_size,
                                                data_b=data_b,
                                                label_b=label_b,
                                                alpha_b=100,
                                                lr = lr,
                                                num_tsteps=50,
                                                num_epoches=Nsteps,
                                                device=device,
                                                args=args,
                                                dU=dU,
                                                xdim=ndim,
                                                vdim=ndim,
                                                checkpoint=10,
                                                adaptive=True,
                                                alpha_beta= 0.9,
                                                beta = 0.3,
                                                valid_data=valid_data,
                                                valid_w=valid_w,
                                                valid_checkpoint=1,
                                                valid_dU = valid_dU)




        # %%
        batch_size = 2**22
        #eta = 10
        lr = 1e-4
        #eta = 1
        #lam = 1
        #kbt = .5
        loss_list,b_loss_list,tot_loss_list,pinn_loss_list,valid_pinn_loss=train_resample(model=q,
                                                data=data,
                                                w=w,
                                                batchsize=batch_size,
                                                data_b=data_b,
                                                label_b=label_b,
                                                alpha_b=1,
                                                lr = lr,
                                                num_tsteps=Nt,
                                                num_epoches=40,
                                                device=device,
                                                args=args,
                                                xdim=ndim,
                                                vdim=ndim,
                                                dU=dU,
                                                checkpoint=10,
                                                adaptive=True,
                                                alpha_beta= 0.9,
                                                beta = 0.3,
                                                valid_data=valid_data,
                                                valid_w=valid_w,
                                                valid_checkpoint=1,
                                                valid_dU = valid_dU)

        fff = f"1d_double_well/model/valid_loss_gamma{gamma}_kbt{kbt}_pinn.txt"
        np.savetxt(fff, valid_pinn_loss)
        t = np.arange(len(valid_pinn_loss))*40
        plt.figure(figsize=(8,6))
        plt.plot(t,valid_pinn_loss)
        plt.yscale("log")
        plt.xlabel("epochs")
        plt.ylabel("Test loss")
        plt.savefig(f"1d_double_well/fig/gamma{gamma}_kbt{kbt}/valid_loss.png")
        plt.clf()



        # %%
        ## initialize

        q_adaptive_false.to(device)

        batch_size = 2**22
        #eta = 10
        lr = 1e-3
        # kbt = 1
        loss_list,b_loss_list,tot_loss_list,pinn_loss_list,valid_pinn_loss=train_resample(model=q_adaptive_false,
                                                data=data,
                                                w=w_adaptive_false,
                                                batchsize=batch_size,
                                                data_b=data_b,
                                                label_b=label_b,
                                                alpha_b=100,
                                                lr = lr,
                                                num_tsteps=50,
                                                num_epoches=Nsteps,
                                                device=device,
                                                args=args,
                                                dU=dU,
                                                xdim=ndim,
                                                vdim=ndim,
                                                checkpoint=10,
                                                adaptive=False,
                                                alpha_beta= 0.9,
                                                beta = 0.3,
                                                valid_data=valid_data,
                                                valid_w=valid_w,
                                                valid_checkpoint=1,
                                                valid_dU = valid_dU)



        # %%
        batch_size = 2**22
        #eta = 10
        lr = 1e-4
        #eta = 1
        #lam = 1
        #kbt = .5
        loss_list,b_loss_list,tot_loss_list,pinn_loss_list,valid_pinn_loss=train_resample(model=q_adaptive_false,
                                                data=data,
                                                w=w_adaptive_false,
                                                batchsize=batch_size,
                                                data_b=data_b,
                                                label_b=label_b,
                                                alpha_b=1,
                                                lr = lr,
                                                num_tsteps=Nt,
                                                num_epoches=40,
                                                device=device,
                                                args=args,
                                                xdim=ndim,
                                                vdim=ndim,
                                                dU=dU,
                                                checkpoint=10,
                                                adaptive=False,
                                                alpha_beta= 0.9,
                                                beta = 0.3,
                                                valid_data=valid_data,
                                                valid_w=valid_w,
                                                valid_checkpoint=1,
                                                valid_dU = valid_dU)


        fff = f"1d_double_well/model/valid_loss_gamma{gamma}_kbt{kbt}_pinn.txt"
        np.savetxt(fff, valid_pinn_loss)
        t = np.arange(len(valid_pinn_loss))*40
        plt.figure(figsize=(8,6))
        plt.plot(t,valid_pinn_loss)
        plt.yscale("log")
        plt.xlabel("epochs")
        plt.ylabel("Test loss")
        plt.savefig(f"1d_double_well/fig/gamma{gamma}_kbt{kbt}/valid_loss_adaptive_false.png")
        plt.clf()


        # %%
        q.to(device)
        q_pinn.to(device)

        # %%
        '''
        plt.scatter(simulation[:,0],simulation[:,1],c=simulation[:,2])
        plt.colorbar()
        plt.show()
        '''

        # %%
        model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}.pth'
        config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}.txt'
        save_model(q,model_file,config_file)

        # %%
        model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}_pinn.pth'
        config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}_pinn.txt'
        save_model(q_pinn,model_file,config_file)

        # %%
        model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}_adaptive_false.pth'
        config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}_adaptive_false.txt'
        save_model(q_adaptive_false,model_file,config_file)


