#!/usr/bin/env python
# coding: utf-8

# In[10]:


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

import logging

# Configure logging



# In[11]:


ndim = 2
gamma = 1
kbt = 10
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



logging.basicConfig(
    filename=f'muller_potential/log/gamma{gamma}_kbt{kbt}.log',        # Specify the log file name
    filemode='w',              # Use append mode ('a') or overwrite mode ('w')
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    level=logging.INFO          # Set the logging level
)

# sample 
'''
Nx_sample = 1000
Nv_sample = 1000
'''
N_sample = 10**6
NA = 5000
NB = 5000

batch_size = 2048 #not implement

layers = [2*ndim,8,64,64,8,1]
activ  = 'sigmoid'

alpha_t = 1
T = 200
Nt = int(T/alpha_t)
Nsteps = 20
lr = 1e-3

device = torch.device(
                "cuda:7" if torch.cuda.is_available() else "cpu")


xmin,xmax = -1.5,1.2
ymin,ymax = -.2,2
x1 = (torch.rand(size=(N_sample,1),dtype=torch.float32))*(xmax-xmin)+xmin
x2 = (torch.rand(size=(N_sample,1),dtype=torch.float32))*(ymax-ymin)+ymin
x = torch.cat((x1,x2),dim=1)
potential = MullerPotential()

UU = potential.potential(x).numpy()
x = x[UU<100,:]
N_sample = x.shape[0]
print(f"N_Sample: {N_sample}")


v = torch.randn(size=(N_sample,ndim),dtype = torch.float32)*np.sqrt(kbt)
data = torch.cat((x,v),dim=1)
w = torch.ones(size=(data.shape[0],1),dtype=torch.float32,device = device)
w = w/torch.sum(w)
dU = potential.gradient(data[:,:ndim])


r=0.2
vA = torch.randn(size=(NA,ndim),dtype = torch.float32)*np.sqrt(kbt)
xA = torch.from_numpy(potential.points_in_a(NA,r).astype(np.float32))
xA = torch.cat((xA,vA),dim=1)
vB = torch.randn(size=(NB,ndim),dtype = torch.float32)*np.sqrt(kbt)
xB = torch.from_numpy(potential.points_in_b(NB,r).astype(np.float32))
xB = torch.cat((xB,vB),dim=1)
labelA = 0*torch.ones_like(xA[:,0])
labelB = 1*torch.ones_like(xB[:,0])

data_b = torch.cat((xA,xB),dim=0)
label_b = torch.cat((labelA,labelB),dim=0).unsqueeze(dim=1)
print(label_b.shape)
del xA,xB,labelA,labelB

total_loss_list = []
total_b_loss_list = []
total_pinn_loss_list = []
total_tot_loss_list = []

q = FunctionModel(layer_sizes=layers,activation=activ)
#model_file = f'./model/gamma10_kbt0.5_1I.pth'
#config_file = f'./config/gamma10_kbt0.5_1I.txt'
#q = load_model(model_file,config_file)

logging.info(f'Muller potential with gamma={gamma}, kbt={kbt}')
logging.info(f'NN info: ')
logging.info(f'Layers: {layers},activation: {activ}')
logging.info(f'Number of samples: {N_sample}')
logging.info(f'Using device: {device}')

    


# In[12]:


args['lam'] = .10
args['eta'] = .10
print(device)


# In[13]:


## initialize
#data.requires_grad_(True)
q.to(device)
data = data.to(device)
batch_size = 2**26
subtrain_idx = 0
#eta = 10
lr = 1e-4
adaptive = True
beta = 0.8
alpha_beta = 0.9
# kbt = 1
logging.info(f'Subtraining index: {subtrain_idx}')
logging.info(f'Batch size: {batch_size}')
logging.info(f'Learning rate: {lr}')
logging.info(f'Number of training steps: {Nsteps}')
logging.info(f'Number of time steps: {Nt}')
logging.info(f'Args: {args}')
if adaptive:
    logging.info(f'Adaptive sampling enabled, beta: {beta}, alpha_beta: {alpha_beta}')

loss_list,b_loss_list,tot_loss_list,pinn_loss_list=train_resample(model=q,
                                          data=data,
                                          w=w,
                                          batchsize=batch_size,
                                          data_b=data_b,
                                          label_b=label_b,
                                          alpha_b=100,
                                          lr = lr,
                                          num_tsteps=Nt,
                                          num_epoches=Nsteps,
                                          device=device,
                                          args=args,
                                          xdim=ndim,
                                          vdim=ndim,
                                          dU=dU,
                                          checkpoint=10,
                                          adaptive=adaptive,
                                          beta=beta,
                                          alpha_beta = alpha_beta)
total_loss_list += loss_list
total_b_loss_list += b_loss_list
total_pinn_loss_list += pinn_loss_list
total_tot_loss_list += tot_loss_list


# In[14]:


t = np.arange(len(loss_list))  # Time values  


# Create a figure with 3 subplots  
fig, axs = plt.subplots(4, 1, figsize=(10, 20))  

# Plot training loss  
axs[0].plot(t, loss_list, label='Training Loss', color='blue')  
axs[0].set_title('Training Loss vs Time')  
axs[0].set_xlabel('Time')  
axs[0].set_ylabel('Loss')  
axs[0].legend()  
axs[0].grid()  

# Plot batch loss  
axs[1].plot(t, b_loss_list, label='Batch Loss', color='orange')  
axs[1].set_title('Boundary Loss vs Time')  
axs[1].set_xlabel('Time')  
axs[1].set_ylabel('Loss')  
axs[1].legend()  
axs[1].grid()  

# Plot total loss  
axs[2].plot(t, tot_loss_list, label='Total Loss', color='green')  
axs[2].set_title('Total Loss vs Time')  
axs[2].set_xlabel('Time')  
axs[2].set_ylabel('Loss')  
axs[2].legend()  
axs[2].grid()  

axs[3].plot(t, pinn_loss_list, label='Pinn Loss', color='red')  
axs[3].set_title('Pinn Loss vs Time')  
axs[3].set_xlabel('Time')  
axs[3].set_ylabel('Loss')  
axs[3].legend()  
axs[3].grid() 

# Adjust layout  
plt.savefig(f'muller_potential/fig/loss_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.png',dpi = 300, bbox_inches='tight')
model_file = f'./muller_potential/model/gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.pth'
config_file = f'./muller_potential/config/gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.txt'
save_model(q,model_file,config_file)

# In[15]:





# In[16]:


q.to(device)
data = data.to(device)
batch_size = 2**22
#eta = 10
lr = 5e-5
#eta = 1
#lam = 1
#kbt = .5
subtrain_idx += 1
NNsteps = Nsteps 
NNt = Nt * 5 
adaptive = True
beta = 0.8
alpha_beta = 0.9
args['lam'] = 1
args['eta'] = 1

logging.info(f'Subtraining index: {subtrain_idx}')
logging.info(f'Batch size: {batch_size}')
logging.info(f'Learning rate: {lr}')
logging.info(f'Number of training steps: {NNsteps}')
logging.info(f'Number of time steps: {NNt}')
logging.info(f'Args: {args}')
if adaptive:
    logging.info(f'Adaptive sampling enabled, beta: {beta}, alpha_beta: {alpha_beta}')
loss_list,b_loss_list,tot_loss_list,pinn_loss_list=train_resample(model=q,
                                          data=data,
                                          w=w,
                                          batchsize=batch_size,
                                          data_b=data_b,
                                          label_b=label_b,
                                          alpha_b=100,
                                          lr = lr,
                                          num_tsteps=NNt,
                                          num_epoches=Nsteps,
                                          device=device,
                                          args=args,
                                          dU=dU,
                                          checkpoint=10,
                                          xdim=ndim,
                                          vdim=ndim,
                                          adaptive=True,
                                          beta=beta,
                                          alpha_beta = alpha_beta)


# In[17]:


t = np.arange(len(loss_list))  # Time values  


# Create a figure with 3 subplots  
fig, axs = plt.subplots(4, 1, figsize=(10, 20))  

# Plot training loss  
axs[0].plot(t, loss_list, label='Training Loss', color='blue')  
axs[0].set_title('Training Loss vs Time')  
axs[0].set_xlabel('Time')  
axs[0].set_ylabel('Loss')  
axs[0].legend()  
axs[0].grid()  

# Plot batch loss  
axs[1].plot(t, b_loss_list, label='Batch Loss', color='orange')  
axs[1].set_title('Boundary Loss vs Time')  
axs[1].set_xlabel('Time')  
axs[1].set_ylabel('Loss')  
axs[1].legend()  
axs[1].grid()  

# Plot total loss  
axs[2].plot(t, tot_loss_list, label='Total Loss', color='green')  
axs[2].set_title('Total Loss vs Time')  
axs[2].set_xlabel('Time')  
axs[2].set_ylabel('Loss')  
axs[2].legend()  
axs[2].grid()  

axs[3].plot(t, pinn_loss_list, label='Pinn Loss', color='red')  
axs[3].set_title('Pinn Loss vs Time')  
axs[3].set_xlabel('Time')  
axs[3].set_ylabel('Loss')  
axs[3].legend()  
axs[3].grid() 

total_loss_list += loss_list
total_b_loss_list += b_loss_list
total_pinn_loss_list += pinn_loss_list
total_tot_loss_list += tot_loss_list

# Adjust layout  
plt.savefig(f'muller_potential/fig/loss_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.png',dpi = 300, bbox_inches='tight')
model_file = f'./muller_potential/model/gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.pth'
config_file = f'./muller_potential/config/gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.txt'
save_model(q,model_file,config_file)

# In[18]:




# In[19]:




# In[ ]:


q.to(device)
data = data.to(device)
w = w.to(device)
batch_size = 2**26
#eta = 10
lr = 1e-5
subtrain_idx += 1
subtrain_idx += 1
NNsteps = Nsteps * 3
NNt = Nt * 5 
adaptive = True
beta = 0.8
alpha_beta = 0.9
args['lam'] = 1
args['eta'] = 1

logging.info(f'Subtraining index: {subtrain_idx}')
logging.info(f'Batch size: {batch_size}')
logging.info(f'Learning rate: {lr}')
logging.info(f'Number of training steps: {NNsteps}')
logging.info(f'Number of time steps: {NNt}')
logging.info(f'Args: {args}')
if adaptive:
    logging.info(f'Adaptive sampling enabled, beta: {beta}, alpha_beta: {alpha_beta}')
#eta = 1
#lam = 1
#kbt = .5
loss_list,b_loss_list,tot_loss_list,pinn_loss_list=train_resample(model=q,
                                          data=data,
                                          w=w,
                                          batchsize=batch_size,
                                          data_b=data_b,
                                          label_b=label_b,
                                          alpha_b=100,
                                          lr = lr,
                                          num_tsteps=NNt,
                                          num_epoches=NNsteps,
                                          device=device,
                                          args=args,
                                          dU=dU,
                                          checkpoint=10,
                                          xdim=ndim,
                                          vdim=ndim,
                                          adaptive=True,
                                          beta=beta,
                                          alpha_beta = alpha_beta)




# In[ ]:


# Length of the data  
t = np.arange(len(loss_list))  # Time values  


# Create a figure with 3 subplots  
fig, axs = plt.subplots(4, 1, figsize=(10, 20))  

# Plot training loss  
axs[0].plot(t, loss_list, label='Training Loss', color='blue')  
axs[0].set_title('Training Loss vs Time')  
axs[0].set_xlabel('Time')  
axs[0].set_ylabel('Loss')  
axs[0].legend()  
axs[0].grid()  

# Plot batch loss  
axs[1].plot(t, b_loss_list, label='Batch Loss', color='orange')  
axs[1].set_title('Boundary Loss vs Time')  
axs[1].set_xlabel('Time')  
axs[1].set_ylabel('Loss')  
axs[1].legend()  
axs[1].grid()  

# Plot total loss  
axs[2].plot(t, tot_loss_list, label='Total Loss', color='green')  
axs[2].set_title('Total Loss vs Time')  
axs[2].set_xlabel('Time')  
axs[2].set_ylabel('Loss')  
axs[2].legend()  
axs[2].grid()  

axs[3].plot(t, pinn_loss_list, label='Pinn Loss', color='red')  
axs[3].set_title('Pinn Loss vs Time')  
axs[3].set_xlabel('Time')  
axs[3].set_ylabel('Loss')  
axs[3].legend()  
axs[3].grid() 

total_loss_list += loss_list
total_b_loss_list += b_loss_list
total_pinn_loss_list += pinn_loss_list
total_tot_loss_list += tot_loss_list

# Adjust layout  
plt.savefig(f'muller_potential/fig/loss_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.png',dpi = 300, bbox_inches='tight') 
model_file = f'./muller_potential/model/gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.pth'
config_file = f'./muller_potential/config/gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.txt'
save_model(q,model_file,config_file)


t = np.arange(len(total_loss_list))  # Time values  


# Create a figure with 3 subplots  
fig, axs = plt.subplots(4, 1, figsize=(10, 20))  

# Plot training loss  
axs[0].plot(t, total_loss_list, label='Training Loss', color='blue')  
axs[0].set_title('Training Loss vs Time')  
axs[0].set_xlabel('Time')  
axs[0].set_ylabel('Loss')  
axs[0].legend()  
axs[0].grid()  

# Plot batch loss  
axs[1].plot(t, total_b_loss_list, label='Batch Loss', color='orange')  
axs[1].set_title('Boundary Loss vs Time')  
axs[1].set_xlabel('Time')  
axs[1].set_ylabel('Loss')  
axs[1].legend()  
axs[1].grid()  

# Plot total loss  
axs[2].plot(t, total_tot_loss_list, label='Total Loss', color='green')  
axs[2].set_title('Total Loss vs Time')  
axs[2].set_xlabel('Time')  
axs[2].set_ylabel('Loss')  
axs[2].legend()  
axs[2].grid()  

axs[3].plot(t, total_pinn_loss_list, label='Pinn Loss', color='red')  
axs[3].set_title('Pinn Loss vs Time')  
axs[3].set_xlabel('Time')  
axs[3].set_ylabel('Loss')  
axs[3].legend()  
axs[3].grid() 


# Adjust layout  
plt.savefig(f'muller_potential/fig/loss_gamma{gamma}_kbt{kbt}_total.png',dpi = 300, bbox_inches='tight') 
model_file = f'./muller_potential/model/gamma{gamma}_kbt{kbt}.pth'
config_file = f'./muller_potential/config/gamma{gamma}_kbt{kbt}.txt'
save_model(q,model_file,config_file)

# In[ ]:


q.to(device)

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

ddd = torch.zeros(size=(points.shape[0],2*ndim)).to(device)

ddd[:,:ndim] = torch.from_numpy(points).to(device)
ddd[:,ndim:] = 0
qqq = torch.zeros(size=(ddd.shape[0],1),dtype=torch.float32,device=device)
NNN = 1000
with torch.no_grad():
    for ttt in range(NNN):
        
        ddd[:,ndim:] = torch.randn(size=(1,ndim),device=device)*torch.ones(size=(ddd.shape[0],ndim),device=device)*np.sqrt(kbt)
        #print(ddd.shape)
        temp = q(ddd)
        
        #print(ddd.shape,temp.shape)
        qqq += temp

qqq = qqq/NNN
qqq = qqq.squeeze().to('cpu').detach()
ddd.requires_grad_(True)
y = q(ddd)



# In[ ]:


contour = plt.contour(
        X,
        Y,
        UU, levels=20,
        cmap='turbo')  # 20 contour levels
plt.scatter(points[:, 0], points[:, 1], c=qqq)
plt.title('Muller potential')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()
plt.savefig(f'muller_potential/fig/ave_muller_potential_gamma{gamma}_kbt{kbt}.png',dpi = 300, bbox_inches='tight')


# In[ ]:



# In[ ]:


vs = np.loadtxt(f'./muller_potential/model/simulation_{kbt}/simulation_vconfig_kbt{kbt}.txt')
v_sample = vs.shape[0]


# In[ ]:


mm=5
nn=5
fig, axs = plt.subplots(mm, nn, figsize=(mm*7, nn*5))  
q.to(device)

# Generate random data for each subplot  
for i in range(mm):  
    for j in range(nn):  
        idx = j+nn*i
        vvm1 = vs[idx,:]
        simulation = np.loadtxt(f'./muller_potential/model/simulation_{kbt}/simulation_kbt{kbt}_gamma{gamma}_{idx}_2.txt')
        ddd = np.zeros(shape=(simulation.shape[0],2*ndim),dtype=np.float32)
        ddd[:,:ndim] = simulation[:,:ndim]
        ddd[:,ndim:] = vvm1
        ddd = torch.from_numpy(ddd).to(device)
        ddd.requires_grad_(True)
        qqq = q(ddd).detach().squeeze().cpu().numpy()
        
        # Create scatter plot  
        sc=axs[i, j].scatter(simulation[:,0],simulation[:,1],c=simulation[:,2])  
        axs[i, j].set_title(f'Scatter Plot {idx+1}')  
        axs[i, j].set_xlabel('x1')  
        axs[i, j].set_ylabel('x2')
        fig.colorbar(sc, ax=axs[i,j])


plt.savefig(f'muller_potential/fig/simulation_gamma{gamma}_kbt{kbt}.png',dpi = 300, bbox_inches='tight')


# In[ ]:


mm=5
nn=5
fig, axs = plt.subplots(mm, nn, figsize=(mm*7, nn*5))  
q.to(device)

# Generate random data for each subplot  
for i in range(mm):  
    for j in range(nn):  
        idx = j+nn*i
        vvm1 = vs[idx,:]
        '''
        ddd.requires_grad_(False)
        ddd[:,(ndim):] = vvm1
        ddd.requires_grad_(True)
        dU1 = potential.gradient(ddd[:,:ndim])
        pinn_l = pinn_loss(q(ddd),ddd,dU1,args)
        pinn_l = pinn_l.detach().cpu().numpy()
        qqq1 = q(ddd).squeeze().to('cpu').detach() 
        '''
        simulation = np.loadtxt(f'./muller_potential/model/simulation_{kbt}/simulation_kbt{kbt}_gamma{gamma}_{idx}_2.txt')
        ddd = np.zeros(shape=(simulation.shape[0],2*ndim),dtype=np.float32)
        ddd[:,:ndim] = simulation[:,:ndim]
        ddd[:,ndim:] = vvm1
        ddd = torch.from_numpy(ddd).to(device)
        ddd.requires_grad_(True)
        qqq = q(ddd).detach().squeeze().cpu().numpy()
        
        # Create scatter plot  
        sc=axs[i, j].scatter(simulation[:,0],simulation[:,1],c=qqq)  
        axs[i, j].set_title(f'Scatter Plot {idx+1}')  
        axs[i, j].set_xlabel('x1')  
        axs[i, j].set_ylabel('x2')
        fig.colorbar(sc, ax=axs[i,j])

plt.savefig(f'muller_potential/fig/NN_gamma{gamma}_kbt{kbt}.png',dpi = 300, bbox_inches='tight')

# In[ ]:

mm=5
nn=5
fig, axs = plt.subplots(mm, nn, figsize=(mm*7, nn*5))  
q.to(device)
l2_loss = 0
l2_norm = 0


# Generate random data for each subplot  
for i in range(mm):  
    for j in range(nn):  
        idx = j+nn*i
        vvm1 = vs[idx,:]
        simulation = np.loadtxt(f'muller_potential/model/simulation_{kbt}/simulation_kbt{kbt}_gamma{gamma}_{idx}_2.txt')
        ddd = np.zeros(shape=(simulation.shape[0],2*ndim),dtype=np.float32)
        ddd[:,:ndim] = simulation[:,:ndim]
        ddd[:,ndim:] = vvm1
        ddd = torch.from_numpy(ddd).to(device)
        ddd.requires_grad_(True)
        qqq = q(ddd).detach().squeeze().cpu().numpy()
        
        # Create scatter plot  
        sc=axs[i, j].scatter(simulation[:,0],simulation[:,1],c=np.abs(qqq-simulation[:,2]))  
        axs[i, j].set_title(f'Scatter Plot {idx+1}')  
        axs[i, j].set_xlabel('x')  
        axs[i, j].set_ylabel('v')
        fig.colorbar(sc, ax=axs[i,j])
        MP = MullerPotential()
        U = MP.potential(simulation[:,0:ndim])
        p = np.exp(-(U-min(U))/kbt)
        p = p/np.sum(p)
        l2_loss += np.sum(p*np.abs(qqq-simulation[:,2])**2)
        l2_norm += np.sum(p*simulation[:,2]**2)

l2_loss /= mm*nn
l2_norm /= mm*nn 
logging.info(f'Absolute error: {l2_loss**0.5}')
logging.info(f'Relative error: {l2_loss**0.5/l2_norm**0.5}')
logging.info(f'l2 norm: {l2_norm**0.5}')
plt.savefig(f'muller_potential/fig/error_gamma{gamma}_kbt{kbt}.png',dpi = 300, bbox_inches='tight')



mm=5
nn=5
fig, axs = plt.subplots(mm, nn, figsize=(mm*7, nn*5))  
q.to(device)


# Generate random data for each subplot  
for i in range(mm):  
    for j in range(nn):  
        idx = j+nn*i
        vvm1 = vs[idx,:]
        
        ddd = np.zeros(shape=(simulation.shape[0],2*ndim),dtype=np.float32)
        ddd[:,:ndim] = simulation[:,:ndim]
        ddd[:,ndim:] = vvm1
        ddd = torch.from_numpy(ddd).to(device)
        ddd.requires_grad_(True)
        dU1 = potential.gradient(ddd[:,:ndim])
        pinn_l = pinn_loss(q(ddd),ddd,dU1,args)
        pinn_l = pinn_l.detach().cpu().numpy()
        
        
        # Create scatter plot  
        sc=axs[i, j].scatter(simulation[:,0],simulation[:,1],c=np.abs(pinn_l))  
        axs[i, j].set_title(f'Scatter Plot {idx+1}')  
        axs[i, j].set_xlabel('x1')  
        axs[i, j].set_ylabel('x2')
        fig.colorbar(sc, ax=axs[i,j])

plt.savefig(f'muller_potential/fig/pinn_gamma{gamma}_kbt{kbt}.png',dpi = 300, bbox_inches='tight')

# In[ ]:


#simulation = np.loadtxt('./model/simulation_kbt.1_gamma10.txt')
q0 = np.loadtxt(f'./muller_potential/model/fd_kbt{kbt}.txt')
#q_simulation = np.loadtxt('./model/q_s_1d.txt')


# In[ ]:


'''
plt.scatter(simulation[:,0],simulation[:,1],c=simulation[:,2])
plt.colorbar()
plt.show()
'''



# In[ ]:


ddd = torch.zeros(size=(q0.shape[0],2*ndim)).to(device)

ddd[:,:ndim] = torch.from_numpy(q0[:,:ndim]).to(device)
ddd[:,ndim:] = 0
qqq = torch.zeros(size=(ddd.shape[0],1),dtype=torch.float32,device=device)
NNN = 1000
with torch.no_grad():
    for ttt in range(NNN):
        
        ddd[:,ndim:] = torch.randn(size=(1,ndim),device=device)*torch.ones(size=(ddd.shape[0],ndim),device=device)*np.sqrt(kbt)
        #print(ddd.shape)
        temp = q(ddd)
        
        #print(ddd.shape,temp.shape)
        qqq += temp

qqq = qqq/NNN
qqq = qqq.squeeze().to('cpu').detach()



# In[ ]:



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
# print(X.shape, V)

points = np.array([X.reshape(-1), Y.reshape(-1)]).T.astype(np.float32)
'''
c = np.arange(len(points))
plt.scatter(points[:, 0], points[:, 1], c=c)
plt.colorbar()
plt.show()
'''
N_matrix = Nrow * Ncol

ddd = torch.zeros(size=(points.shape[0],2*ndim)).to(device)

ddd[:,:ndim] = torch.from_numpy(points).to(device)
ddd[:,ndim:] = 0


# In[ ]:


ttt = np.abs(q0[:,2]-qqq.numpy())
#ttt[ttt>0.3] = 0.3
plt.scatter(q0[:,0], q0[:,1], c=ttt)
plt.title('$|q_0^{ref} - q_0^{NN}|$')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()
plt.savefig(f'muller_potential/fig/ave_error_gamma{gamma}_kbt{kbt}.png',dpi = 300, bbox_inches='tight')


# In[ ]:


MP = MullerPotential()
U = MP.potential(q0[:,:ndim])
p = np.exp(-(U-min(U))/kbt)
p = p/np.sum(p)
print(f'Absolute error: {np.sum(p*ttt**2)**0.5}')
print(f'Reletive error: {np.sum(p*ttt**2)**0.5/np.sum(p*q0[:,2]**2)**0.5}')
print(f'l2 norm: {np.sum(p*q0[:,2]**2)**0.5}')


# In[ ]:


'''
plt.scatter(points[:, 0], points[:, 1], c=qqq)
plt.title('committor $q(x,v)$')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()
plt.show()
'''


# In[ ]:


'''
vslice = 0.0
Q = qqq.reshape(X.shape)
Qfd = fd[:,2].reshape(X.shape)
#plt.plot(xcal[::10], Q[int((vslice-vmin)/dv), ::10]-Qfd[int((vslice-vmin)/dv), ::10])
plt.plot(xcal[:], Q[int((vslice-vmin)/dv), :]-Qfd[int((vslice-vmin)/dv), :])
plt.plot(xcal[:], Q[int((vslice-vmin)/dv), :],'r')
plt.plot(xcal[:], q0,'g')
plt.plot(xcal[:],Qfd[int((vslice-vmin)/dv), :],'b')
#plt.plot(xcal, Q[int((vslice-vmin)/dv), :])
plt.xlabel('x1')
plt.ylabel('q')
plt.title(f'slice with v={vslice}')
plt.show()
'''




# In[ ]:


model_file = f'./muller_potential/model/gamma{gamma}_kbt{kbt}.pth'
config_file = f'./muller_potential/config/gamma{gamma}_kbt{kbt}.txt'
save_model(q,model_file,config_file)

