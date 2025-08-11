
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

import logging

# Configure logging

ndim = 2
gamma = 25
kbt = 5
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



model_file = f'./muller_potential/model/gamma{gamma}_kbt{kbt}_subtrain_6.pth'
config_file = f'./muller_potential/config/gamma{gamma}_kbt{kbt}_subtrain_6.txt'
q = load_model(model_file,config_file)

# In[ ]:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
potential = MullerPotential()
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

fig = plt.figure(figsize=(5, 5))
plt.contour(
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

