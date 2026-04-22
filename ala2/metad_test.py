#!/usr/bin/env python
# coding: utf-8

# In[10]:


import sys
import os
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
from model_training import train_resample,pinn_loss,build_rightside, train_mass
from hist import hist_reweight
from utils import *



import logging

# Configure logging



# In[11]:

num_heavy_atoms = heavy_atom_indices.shape[0]
xdim = heavy_atom_indices.shape[0] * 3
vdim = heavy_atom_indices.shape[0] * 3
use_distance = False
if use_distance:
    xdim_reduce = 45
    vdim_reduce = 45
else:
    xdim_reduce = 4
    vdim_reduce = 4


ndim = xdim
gamma = 1 
kbt = 300 * 0.0083144621  # kBT in kcal/mol   
lam = 10
eta = 10
omega = 5


#data_label = 'highT'

biased = True 
bias_decay = 0.9
#data_label = 'long'

data_label = 'biased'
#data_label = 'all'
#data_label = 'all_normalized'
#data_label = 'constrained'

subtrain_idx = 39
mask_AB = False

args = {
        "xdim": xdim,
        "vdim": vdim,
        "gamma": gamma,
        "kbt": kbt,
        "lam": lam,
        "eta": eta,
        "omega": omega,
        "ndim": ndim,
        "xdim_reduce": xdim_reduce,
        "vdim_reduce": vdim_reduce
    }



logging.basicConfig(
    filename=f'ala2/log/gamma{gamma}_kbt{kbt}_{data_label}.log',        # Specify the log file name
    filemode='w',              # Use append mode ('a') or overwrite mode ('w')
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    level=logging.INFO          # Set the logging level
)

# sample 
'''
Nx_sample = 1000
Nv_sample = 1000
'''


batch_size = 2**26 #not implement

#layers = [xdim_reduce+vdim_reduce,8,64,64,64,64,8,1]
layers = [xdim_reduce+vdim_reduce,8,256,256,1]
activ  = 'sigmoid'

alpha_t = 1
T = 200
Nt = int(T/alpha_t)
Nsteps = 40
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


highT_path = "ala2/simulation/1500K/"  # Working directory for intermediate files
C7eq_path = "ala2/simulation/long_C7eq/"
C7ax_path = "ala2/simulation/long_C7ax/"
long_C7eq_path = "ala2/simulation/long_C7eq/"
long_C7ax_path = "ala2/simulation/long_C7ax/"
#bias_C7eq_path = f"ala2/simulation/biased_gamma{gamma}_highT/long_C7eq/"
#bias_C7ax_path = f"ala2/simulation/biased_gamma{gamma}_highT/long_C7ax/"

positions_filename = "positions.xvg"
velocities_filename = "velocities.xvg"
forces_filename = "forces.xvg"

def load_data(file_path,positions_filename,velocities_filename,forces_filename,heavy_atom_indices):

#extract_trr_data(trr_file, tpr_file,C7ax_path,positions_filename,velocities_filename,forces_filename)
    C7eq_xs = read_xvg(file_path+positions_filename)
    C7eq_vs = read_xvg(file_path+velocities_filename)
    C7eq_fs = read_xvg(file_path+forces_filename)
    C7eq_xv,C7eq_fs,C7eq_xv_heavy,C7eq_fs_heavy = preprocessing_data_np2torch(C7eq_xs,C7eq_vs,C7eq_fs,heavy_atom_indices)
    return C7eq_xv,C7eq_fs,C7eq_xv_heavy,C7eq_fs_heavy



metad_path = "ala2/simulation/metad/"
_,_,s_C7eq_xv_heavy,s_C7eq_fs_heavy = load_data(metad_path,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
s_C7eq_xv_heavy.requires_grad_(True)
data_metad_x = s_C7eq_xv_heavy[:,:xdim]
colvar_name = 'COLAR_METAD_reweight_1'
col_C7eq = read_COLVAR(metad_path+colvar_name)
col_C7eq = torch.from_numpy(col_C7eq)
hills = np.loadtxt('ala2/simulation/metad/HILLS')

dd_metad = np.loadtxt('ala2/simulation/metad/dmetad')
dphipsi =dd_metad[:,2].reshape(-1,2)
dphipsi = torch.from_numpy(dphipsi.astype(np.float32)).to(device)

phipsi_metad = phipsi(data_metad_x, num_heavy_atoms, phi_group_heavy, psi_group_heavy)
y = dphipsi*phipsi_metad.to(device)
x_force = torch.autograd.grad(outputs=y, 
                                inputs=data_metad_x,
                                grad_outputs=torch.ones_like(y),
                                create_graph=False, retain_graph=False)[0]



