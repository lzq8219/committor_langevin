#!/usr/bin/env python
# coding: utf-8

# In[10]:


import sys
import os
import subprocess

origin_directory = os.getcwd()
model_directory = os.path.join(origin_directory, 'ala2')
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
from model_training import train_resample,pinn_loss,build_rightside, train_mass,train_overdamped
from hist import hist_reweight
from diagnose import draw_q0_dq0
from utils import *

import logging

# Configure logging



# In[11]:

num_heavy_atoms = heavy_atom_indices.shape[0]
xdim = heavy_atom_indices.shape[0] * 3
xdim_reduce = 4

#xdim_reduce = 45

ndim = xdim
gamma_0 = 10300
itrs = 10
kbt = 300 * 0.0083144621  # kBT in kcal/mol   
lam = 10
eta = 10
omega = gamma_0

continue_training = 0

args = {
        "xdim": xdim,
        "gamma": gamma_0,
        "kbt": kbt,
        "lam": lam,
        "eta": eta,
        "omega": omega,
        "ndim": ndim,
        "xdim_reduce": xdim_reduce,
    }



logging.basicConfig(
    filename=f'ala2/log/overdamped_gamma{gamma_0}_kbt{kbt}_with_itr.log',        # Specify the log file name
    filemode='w',              # Use append mode ('a') or overwrite mode ('w')
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    level=logging.INFO          # Set the logging level
)

# sample 
'''
Nx_sample = 1000
Nv_sample = 1000
'''


batch_size = 2048 #not implement

layers = [xdim_reduce,8,64,64,8,1]
activ  = 'sigmoid'

alpha_t = 1
T = 200
Nt = int(T/alpha_t)
Nsteps = 60
lr = 1e-3

device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")



# loading data topology_file = "ala2/simulation/topol.top"
topology_file = "ala2/simulation/topol.top"
mass = read_mass(topology_file)
mass=torch.tensor(mass)
heavy_atom_mass = mass[heavy_atom_indices]
mass = torch.repeat_interleave(mass,3).unsqueeze(0)
heavy_atom_mass = torch.repeat_interleave(heavy_atom_mass,3).unsqueeze(0)
num_heavy_atoms = heavy_atom_indices.shape[0]

print('mass:',mass)
print('heavy_atom_mass:',heavy_atom_mass)


highT_path = "ala2/simulation/400K/"  # Working directory for intermediate files
C7eq_path = "ala2/simulation/long_C7eq/"
C7ax_path = "ala2/simulation/long_C7ax/"
long_C7eq_path = "ala2/simulation/long_C7eq/"
long_C7ax_path = "ala2/simulation/long_C7ax/"

positions_filename = "positions.xvg"

#extract_trr_data(trr_file, tpr_file,C7ax_path,positions_filename,velocities_filename,forces_filename)
C7eq_xs = read_xvg(C7eq_path+positions_filename)
C7eq_xs = torch.from_numpy(C7eq_xs[:int(C7eq_xs.shape[0]),heavy_dim_indices].astype(np.float32))


long_C7eq_xs = read_xvg(long_C7eq_path+positions_filename)
long_C7eq_xs = torch.from_numpy(long_C7eq_xs[:,heavy_dim_indices].astype(np.float32))

C7ax_xs = read_xvg(C7ax_path+positions_filename)
C7ax_xs = torch.from_numpy(C7ax_xs[:int(C7ax_xs.shape[0]),heavy_dim_indices].astype(np.float32))

long_C7ax_xs = read_xvg(long_C7ax_path+positions_filename)
long_C7ax_xs = torch.from_numpy(long_C7ax_xs[:,heavy_dim_indices].astype(np.float32))
#bias_C7ax = read_COLVAR(long_C7ax_path+"COLVAR_1")[:,3]
#bias_C7ax = torch.from_numpy(bias_C7ax.astype(np.float32))
#w_C7ax = torch.exp(kbt**-1 * bias_C7ax)
#w_C7ax = w_C7ax/torch.sum(w_C7ax)
part = 0.1
data_A = C7eq_xs[:int(C7eq_xs.shape[0]*part),:]
data_B = C7ax_xs[:int(C7ax_xs.shape[0]*part),:]
label_a = torch.zeros(data_A.shape[0], dtype=torch.float32)
label_b = torch.ones(data_B.shape[0], dtype=torch.float32)
data_boundary = torch.cat((data_A,data_B), dim=0)
label_boundary = torch.cat((label_a, label_b), dim=0).unsqueeze(1)

heavy_atom_mass = torch.ones_like(heavy_atom_mass).to(device)



itr = continue_training



data = torch.cat((long_C7eq_xs,long_C7ax_xs), dim=0).to(device)
#w = torch.cat((w_C7eq,w_C7ax), dim=0).unsqueeze(1)

#w1 = read_COLVAR(long_C7eq_path+"COLVAR_3")[:,3]
#w1 = np.exp(kbt**-1 * w1)
#w1 = w1/np.sum(w1)
#w2 = read_COLVAR(long_C7ax_path+"COLVAR_3")[:,3]
#w2 = np.exp(kbt**-1 * w2)
#w2 = w2/np.sum(w2)
#w = torch.from_numpy(np.concatenate((w1,w2),axis=0).astype(np.float32)).unsqueeze(1)
wu = torch.ones(data.shape[0], dtype=torch.float32).unsqueeze(1)
wu = wu/torch.sum(wu)
w = wu
w=w.to(device)
wu = wu.to(device)
l=-2
for ii in range(itr+1):
    if ii == 0:
        continue
    i = ii+200

    filename = f'./o_{i}/bias.sh'
    plumed_file = f'./o_{i}/plumed_q0.dat'
    itr_path = f'./o_{i}'
    long_C7ax_path_itr = f'./o_{i}/long_C7ax_{l}'
    long_C7eq_path_itr = f'./o_{i}/long_C7eq_{l}'

    os.chdir('ala2/simulation')

    

    long_C7eq_xs = read_xvg(long_C7eq_path_itr+'/'+positions_filename)
    long_C7eq_xs = torch.from_numpy(long_C7eq_xs[:,heavy_dim_indices].astype(np.float32)).to(device)


    long_C7ax_xs = read_xvg(long_C7ax_path_itr+'/'+positions_filename)
    long_C7ax_xs = torch.from_numpy(long_C7ax_xs[:,heavy_dim_indices].astype(np.float32)).to(device)

    data = torch.cat((data,long_C7eq_xs,long_C7ax_xs), dim=0)

    w1 = read_COLVAR(long_C7eq_path_itr+'/'+f"COLVAR_{l}")[:,3]
    w1 = np.exp(kbt**-1 * w1)
    w1 = w1/np.sum(w1)
    w2 = read_COLVAR(long_C7ax_path_itr+'/'+f"COLVAR_{l}")[:,3]
    w2 = np.exp(kbt**-1 * w2)
    w2 = w2/np.sum(w2)
    ww = torch.from_numpy(np.concatenate((w1,w2),axis=0).astype(np.float32)).unsqueeze(1).to(device)
    print(w.shape,ww.shape)
    w = torch.cat((w,ww),dim = 0)


    os.chdir("../..")

if itr == 0:
        q_old = None
else:
    model_file_old = f'./ala2/model/o_gamma{gamma_0+itr-1}_kbt{kbt}.pth'
    config_file_old = f'./ala2/config/o_gamma{gamma_0+itr-1}_kbt{kbt}.txt'
    q_old  = load_model_phipsi(model_file_old,config_file_old,layers)
    print(q_old.output_scale)
    q_old.to(device)
    q_old.eval()

while True:

    total_loss_list = []
    total_b_loss_list = []
    total_pinn_loss_list = []
    total_tot_loss_list = []

    
    gamma = gamma_0 + itr
    
    
    q = NNphipsi_overdamped(layer_sizes=layers,n_atoms=num_heavy_atoms, activation='sigmoid',phi_group=phi_group_heavy,psi_group=theta_group_heavy,output_scale=itr+1)

    

    #model_file = f'./model/gamma10_kbt0.5_1I.pth'
    #config_file = f'./config/gamma10_kbt0.5_1I.txt'
    #q = load_model(model_file,config_file)

    logging.info(f'Potential with gamma={gamma}, kbt={kbt}')
    logging.info(f'NN info: ')
    logging.info(f'Layers: {layers},activation: {activ}')
    logging.info(f'Number of samples: {data.shape[0]}')
    logging.info(f'Using device: {device}')

        


    # In[12]:



    # In[13]:


    ## initialize
    #data.requires_grad_(True)
    q.to(device)
    data = data.to(device)


    batch_size = 2**26
    subtrain_idx = 0
    #eta = 10
    lr = 5e-4
    adaptive = True
    beta = 0.8
    alpha_beta = 0.9
    pinn_weight = 0.9 
    grad_weight = 0.05
    NNt = Nt *5
    NNsteps = Nsteps * 1
    # kbt = 1
    logging.info(f'Subtraining index: {subtrain_idx}')
    logging.info(f'Batch size: {batch_size}')
    logging.info(f'Learning rate: {lr}')
    logging.info(f'Number of training steps: {NNsteps}')
    logging.info(f'Number of time steps: {NNt}')
    logging.info(f'Args: {args}')
    if adaptive:
        logging.info(f'Adaptive sampling enabled, beta: {beta}, pinn_weight: {pinn_weight}, grad_weight: {grad_weight}')



    loss_list,b_loss_list,tot_loss_list=train_overdamped(model=q,
                                            data=data,
                                            mass = heavy_atom_mass,
                                            w=w,
                                            batchsize=batch_size,
                                            data_b=data_boundary,
                                            label_b=label_boundary,
                                            alpha_b=10000,
                                            lr = 1e-3,
                                            num_tsteps=Nt*5,
                                            num_epoches=Nsteps,
                                            device=device,
                                            args=args,
                                            xdim=ndim,
                                            vdim=ndim,
                                            checkpoint=10,
                                            alpha_l2 = 1e-7,
                                            model_old = q_old)


    loss_list,b_loss_list,tot_loss_list=train_overdamped(model=q,
                                            data=data,
                                            mass = heavy_atom_mass,
                                            w=w,
                                            batchsize=batch_size,
                                            data_b=data_boundary,
                                            label_b=label_boundary,
                                            alpha_b=10000,
                                            lr = lr,
                                            num_tsteps=NNt,
                                            num_epoches=NNsteps,
                                            device=device,
                                            args=args,
                                            xdim=ndim,
                                            vdim=ndim,
                                            checkpoint=10,
                                            alpha_l2 = 1e-7,
                                            model_old = q_old)
    total_loss_list += loss_list
    total_b_loss_list += b_loss_list
    total_tot_loss_list += tot_loss_list


    # In[14]:


    t = np.arange(len(loss_list))  # Time values  


    # Create a figure with 3 subplots  
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))  

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



    # Adjust layout  
    plt.savefig(f'ala2/fig/o_loss_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.png',dpi = 300, bbox_inches='tight')
    model_file = f'./ala2/model/o_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.pth'
    config_file = f'./ala2/config/o_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.txt'
    q.save(data,config_file,model_file)
    plt.close()

    # In[15]:





    # In[16]:


    q.to(device)
    data = data.to(device)
    batch_size = 2**26
    #eta = 10
    lr = 1e-4
    #eta = 1
    #lam = 1
    #kbt = .5
    subtrain_idx += 1
    NNsteps = Nsteps *3
    NNt = Nt * 5
    adaptive = True
    beta = 0.8
    alpha_beta = 0.95
    pinn_weight = 0.9 
    grad_weight = 0.05
    args['lam'] = 5000
    args['eta'] = 5000


    logging.info(f'Subtraining index: {subtrain_idx}')
    logging.info(f'Batch size: {batch_size}')
    logging.info(f'Learning rate: {lr}')
    logging.info(f'Number of training steps: {NNsteps}')
    logging.info(f'Number of time steps: {NNt}')
    logging.info(f'Args: {args}')
    if adaptive:
        logging.info(f'Adaptive sampling enabled, beta: {beta}, pinn_weight: {pinn_weight}, grad_weight: {grad_weight}')
    loss_list,b_loss_list,tot_loss_list=train_overdamped(model=q,
                                            data=data,
                                            mass = heavy_atom_mass,
                                            w=w,
                                            batchsize=batch_size,
                                            data_b=data_boundary,
                                            label_b=label_boundary,
                                            alpha_b=10000,
                                            lr = lr,
                                            num_tsteps=NNt,
                                            num_epoches=NNsteps,
                                            device=device,
                                            args=args,
                                            checkpoint=10,
                                            xdim=ndim,
                                            vdim=ndim,
                                            alpha_l2 = 1e-7,
                                            model_old = q_old)


    # In[17]:


    t = np.arange(len(loss_list))  # Time values  


    # Create a figure with 3 subplots  
    fig, axs = plt.subplots(3, 1, figsize=(15,15))  

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


    total_loss_list += loss_list
    total_b_loss_list += b_loss_list
    total_tot_loss_list += tot_loss_list

    # Adjust layout  
    plt.savefig(f'ala2/fig/o_loss_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.png',dpi = 300, bbox_inches='tight')
    model_file = f'./ala2/model/o_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.pth'
    config_file = f'./ala2/config/o_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.txt'
    q.save(data,config_file,model_file)
    plt.close()

    # In[18]:




    # In[19]:




    # In[ ]:


    q.to(device)
    data = data.to(device)
    w = w.to(device)
    batch_size = 2**26
    #eta = 10
    lr = 5e-5
    subtrain_idx += 1
    NNsteps = Nsteps * 3
    NNt = Nt * 5 
    adaptive = True
    beta = 0.8
    alpha_beta = 0.99
    pinn_weight = 0.9 
    grad_weight = 0.05
    args['lam'] = 5000
    args['eta'] = 5000

    logging.info(f'Subtraining index: {subtrain_idx}')
    logging.info(f'Batch size: {batch_size}')
    logging.info(f'Learning rate: {lr}')
    logging.info(f'Number of training steps: {NNsteps}')
    logging.info(f'Number of time steps: {NNt}')
    logging.info(f'Args: {args}')
    if adaptive:
        logging.info(f'Adaptive sampling enabled, beta: {beta}, pinn_weight: {pinn_weight}, grad_weight: {grad_weight}')
    #eta = 1
    #lam = 1
    #kbt = .5
    loss_list,b_loss_list,tot_loss_list=train_overdamped(model=q,
                                            data=data,
                                            mass = heavy_atom_mass,
                                            w=w,
                                            batchsize=batch_size,
                                            data_b=data_boundary,
                                            label_b=label_boundary,
                                            alpha_b=10000,
                                            lr = lr,
                                            num_tsteps=NNt,
                                            num_epoches=NNsteps,
                                            device=device,
                                            args=args,
                                            checkpoint=10,
                                            xdim=ndim,
                                            vdim=ndim,
                                            alpha_l2 = 1e-7,
                                            model_old = q_old)




    # In[ ]:


    # Length of the data  
    t = np.arange(len(loss_list))  # Time values  


    # Create a figure with 3 subplots  
    fig, axs = plt.subplots(3, 1, figsize=(15,15))  

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


    

    total_loss_list += loss_list
    total_b_loss_list += b_loss_list
    total_tot_loss_list += tot_loss_list

    # Adjust layout  
    plt.savefig(f'ala2/fig/o_loss_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.png',dpi = 300, bbox_inches='tight') 
    model_file = f'./ala2/model/o_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.pth'
    config_file = f'./ala2/config/o_gamma{gamma}_kbt{kbt}_subtrain_{subtrain_idx}.txt'
    q.save(data,config_file,model_file)


    t = np.arange(len(total_loss_list))  # Time values  


    # Create a figure with 3 subplots  
    fig, axs = plt.subplots(3, 1, figsize=(15,15))  

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


    # Adjust layout  
    plt.savefig(f'ala2/fig/o_loss_gamma{gamma}_kbt{kbt}_total.png',dpi = 300, bbox_inches='tight') 
    model_file = f'./ala2/model/o_gamma{gamma}_kbt{kbt}.pth'
    config_file = f'./ala2/config/o_gamma{gamma}_kbt{kbt}.txt'
    q.save(data,config_file,model_file)
    plt.close()
    itr += 1
    if itr >itrs:
        break

    filename = f'./o_{itr+200}/simulation_input.txt'
    itr_path = f'./o_{itr+200}'
    long_C7ax_path_itr = f'./o_{itr+200}/long_C7ax_{l}'
    long_C7eq_path_itr = f'./o_{itr+200}/long_C7eq_{l}'
    model_file = f'../model/o_gamma{gamma}_kbt{kbt}.pth'
    os.chdir('ala2/simulation')
    if not os.path.exists(itr_path):
        os.makedirs(itr_path)

    with open(filename, 'w') as f:
        content = f'''em_C7eq.tpr plumed_C7eq_{l}.dat {model_file} {l} COLVAR_{l} {long_C7eq_path_itr}
em_C7ax.tpr plumed_C7ax_{l}.dat {model_file} {l} COLVAR_{l} {long_C7ax_path_itr}'''
        f.write(content)
    _cmd = " parallel -j10 --env PIN_CORES --env OMP_NUM_THREADS --joblog parallel.log 'PARALLEL_SLOT={%} ./biased_simulation.sh {1} {2} {3} {4} {5} {6}'"
    cmd0 = f"cat {filename} |"+_cmd
    print(cmd0)
    subprocess.run(cmd0,shell=True)

    long_C7eq_xs = read_xvg(long_C7eq_path_itr+'/'+positions_filename)
    long_C7eq_xs = torch.from_numpy(long_C7eq_xs[:,heavy_dim_indices].astype(np.float32)).to(device)


    long_C7ax_xs = read_xvg(long_C7ax_path_itr+'/'+positions_filename)
    long_C7ax_xs = torch.from_numpy(long_C7ax_xs[:,heavy_dim_indices].astype(np.float32)).to(device)
    data_biased_itr = torch.cat((long_C7eq_xs,long_C7ax_xs), dim=0)


    data = torch.cat((data,long_C7eq_xs,long_C7ax_xs), dim=0)

    w1 = read_COLVAR(long_C7eq_path_itr+'/'+f"COLVAR_{l}")[:,3]
    w1 = np.exp(kbt**-1 * w1)
    w1 = w1/np.sum(w1)
    w2 = read_COLVAR(long_C7ax_path_itr+'/'+f"COLVAR_{l}")[:,3]
    w2 = np.exp(kbt**-1 * w2)
    w2 = w2/np.sum(w2)
    ww = torch.from_numpy(np.concatenate((w1,w2),axis=0).astype(np.float32)).unsqueeze(1).to(device)
    print(w.shape,ww.shape)
    wu = torch.cat((wu,ww),dim = 0)
    w = wu/torch.sum(wu)


    os.chdir("../..")
    figname_q0 = f'ala2/fig/q0_{itr}_{l}_summation.png'
    filename_dq0 = f'ala2/fig/dq0_{itr}_{l}_summation.png'
    plt.clf()
    
    draw_q0_dq0(q,360,data_biased_itr,figname_q0,filename_dq0,device=device,q_old = q_old)
    q_old = q
    q_old.eval()

data = data.detach().cpu().numpy()
np.savetxt(f"ala2/simulation/data_gamma_{gamma_0}kbt_{kbt}_{l}.txt", data)
# In[ ]:





