# %%
import sys
import os
parent_directory = os.path.dirname(os.getcwd())
src_directory = os.path.join(parent_directory, 'src')
sys.path.append(src_directory)
sys.path.append('src')
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from nn import FunctionModel, save_model, load_model
import copy
import matplotlib.pyplot as plt
from model_training import train_resample,pinn_loss,build_rightside,train_pinn,train_HSS


# %%
ndim = 3
gamma = 0.2
kbt = .1
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

del xA,xB,labelA,labelB

q = FunctionModel(layer_sizes=layers,activation=activ)
q_HSS = FunctionModel(layer_sizes=layers,activation=activ)
q_adaptive_false = FunctionModel(layer_sizes=layers,activation=activ)
load_pinn = False
#model_file = f'./model/gamma10_kbt0.5_1I.pth'
#config_file = f'./config/gamma10_kbt0.5_1I.txt'
#q = load_model(model_file,config_file)

if load_pinn:
    model_file = f'./model/gamma{gamma}_kbt{kbt}_pinn.pth'
    config_file = f'./config/gamma{gamma}_kbt{kbt}_pinn.txt'
    q_pinn = load_model(model_file,config_file)


    

# %%


# %%
print(device)

# %%


# %%
args['lam'] = .2
args['eta'] = 0
args['omega'] = gamma

# %%
q_HSS.to(device)

batch_size = 2**22
#eta = 10
lr = 1e-3
lr_HSS_2 = 1e-3
num_epoches_HSS_2 = Nsteps
Nt_steps_checkpoint = 1
q.to(device)
q_HSS.to(device)
for i in range(20):
    train_HSS(model =q_HSS, 
        data = data, 
        w = w, 
        batchsize = batch_size, 
        data_b = data_b, 
        label_b = label_b, 
        dU = dU,
        alpha_b = 100, 
        lr = lr, 
        num_tsteps = Nt_steps_checkpoint,
        num_epoches = Nsteps, 
        device = device, 
        args = args, 
        xdim = ndim, 
        vdim = ndim, 
        checkpoint=10, 
        alpha_l2=0, 
        valid_data=valid_data,
        valid_w=valid_w,
        valid_checkpoint=50,
        valid_dU = valid_dU,
        alpha_char=0,
        lr_HSS_2=lr_HSS_2,
        num_epoches_HSS_2=num_epoches_HSS_2,
        dU_func=dU_func,
        dt=0.001)
    
    train_resample(model=q,
        data=data,
        w=w,
        batchsize=batch_size,
        data_b=data_b,
        label_b=label_b,
        alpha_b=100,
        lr = lr,
        num_tsteps=Nt_steps_checkpoint,
        num_epoches=Nsteps*2,
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
        valid_checkpoint=50,
        valid_dU = valid_dU)
    
    model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}_HSS_step_{i}.pth'
    config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}_HSS_step_{i}.txt'
    save_model(q_HSS,model_file,config_file)

    # %%
    model_file = f'1d_double_well/model/gamma{gamma}_kbt{kbt}_test_step_{i}.pth'
    config_file = f'1d_double_well/config/gamma{gamma}_kbt{kbt}_test_step_{i}.txt'
    save_model(q,model_file,config_file)





# %%



