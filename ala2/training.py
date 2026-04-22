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
from diagnose import quick_diagnose,phi_contour_2, theta_contour_2, U_phitheta
from distilling import distilling_models_phipsi


import logging

# Configure logging



# In[11]:

num_heavy_atoms = heavy_atom_indices.shape[0]
xdim = heavy_atom_indices.shape[0] * 3
vdim = heavy_atom_indices.shape[0] * 3
use_distance = True
if use_distance:
    xdim_reduce = 45
    vdim_reduce = 45
else:
    xdim_reduce = 4
    vdim_reduce = 4


ndim = xdim
gamma = 25
kbt = 300 * 0.0083144621  # kBT in kcal/mol   
lam = 10
eta = 10
omega = gamma


#data_label = 'highT'

biased = True 
bias_decay = 0.9
#data_label = 'long'

#data_label = 'biased'
#data_label = 'all'
#data_label = 'all_normalized'
#data_label = 'constrained'
#data_label = 'metad_1'
data_label = 'biased_1'
#data_label = 'biased_2'

subtrain_idx = 0
mask_AB = True

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

C7eq_xv,C7eq_fs,C7eq_xv_heavy,C7eq_fs_heavy = load_data(C7eq_path,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
C7ax_xv,C7ax_fs,C7ax_xv_heavy,C7ax_fs_heavy = load_data(C7ax_path,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
highT_xv,highT_fs,highT_xv_heavy,highT_fs_heavy = load_data(highT_path,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
long_C7eq_xv,long_C7eq_fs,long_C7eq_xv_heavy,long_C7eq_fs_heavy = load_data(long_C7eq_path,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
long_C7ax_xv,long_C7ax_fs,long_C7ax_xv_heavy,long_C7ax_fs_heavy = load_data(long_C7ax_path,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)


itrs = 6
data_biased = torch.tensor([]).to(device)
dU_biased = torch.tensor([]).to(device)
cv_force = torch.tensor([]).to(device)
w_biased = torch.tensor([]).to(device)
for i in range(1,itrs+1):

    C7eq_path_biased = f"ala2/simulation/o_{i+100}/long_C7eq_-3.5/"
    C7ax_path_biased = f"ala2/simulation/o_{i+100}/long_C7ax_-3.5/"
    colvar_name = 'COLVAR_-3.5'
    _,_,s_C7eq_xv_heavy,s_C7eq_fs_heavy = load_data(C7eq_path_biased,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
    _,_,s_C7ax_xv_heavy,s_C7ax_fs_heavy = load_data(C7ax_path_biased,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
    col_C7eq = read_COLVAR(C7eq_path_biased+colvar_name)
    col_C7ax = read_COLVAR(C7ax_path_biased+colvar_name)
    w1 = torch.from_numpy(col_C7eq[:,3]).to(device)
    w2 = torch.from_numpy(col_C7ax[:,3]).to(device)
    ww = torch.cat((w1, w2), dim=0).unsqueeze(1)
    ww = torch.exp(ww/kbt)
    ww = ww/torch.sum(ww)
    w_biased = torch.cat((w_biased,ww),dim=0)
    cv_force_C7eq = torch.from_numpy(col_C7eq[:,4:8]).to(device)
    cv_force_C7ax = torch.from_numpy(col_C7ax[:,4:8]).to(device)
    data_biased = torch.cat((data_biased,s_C7eq_xv_heavy.to(device),s_C7ax_xv_heavy.to(device)),dim=0)
    dU_biased = torch.cat((dU_biased,-s_C7eq_fs_heavy.to(device),-s_C7ax_fs_heavy.to(device)),dim=0)
    cv_force = torch.cat((cv_force,cv_force_C7eq,cv_force_C7ax),dim=0)
data_biased.requires_grad_(True)
data_biased_x = data_biased[:,:xdim]
y = cv_force*descriptor_phipsi(data_biased_x,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy)
x_force = torch.autograd.grad(outputs=y, 
                                inputs=data_biased_x,
                                grad_outputs=torch.ones_like(y),
                                create_graph=False, retain_graph=False)[0]
dU_biased = dU_biased + x_force
data_biased = data_biased.detach().to('cpu')
dU_biased = dU_biased.detach().to('cpu')

del x_force,cv_force
torch.cuda.empty_cache()


metad_path = "ala2/simulation/metad/"
_,_,s_C7eq_xv_heavy,s_C7eq_fs_heavy = load_data(metad_path,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
s_C7eq_xv_heavy.requires_grad_(True)
data_metad = s_C7eq_xv_heavy
dU_metad = -s_C7eq_fs_heavy
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
dmetad_x = torch.autograd.grad(outputs=y, 
                                inputs=data_metad_x,
                                grad_outputs=torch.ones_like(y),
                                create_graph=False, retain_graph=False)[0]
dU_metad = dU_metad-dmetad_x


print('Yeah!!')
step = 10
deg_to_rad = np.pi / 180.0                      
deg_to_rad = np.float32(deg_to_rad)
phi_angles = np.arange(-180, 180, step).astype(np.float32)
psi_angles = np.arange(-180, 180, step).astype(np.float32)
phi_mesh, psi_mesh = np.meshgrid(phi_angles, psi_angles)
phi_mesh = torch.from_numpy(phi_mesh * deg_to_rad).to(device)
psi_mesh = torch.from_numpy(psi_mesh * deg_to_rad).to(device)
data_constrained = torch.tensor([]).to(device)
dU_constrained = torch.tensor([]).to(device)
cv_force = torch.tensor([]).to(device)
w_constrained = torch.tensor([]).to(device)
if data_label == 'constrained':
    for i in range(phi_mesh.shape[0]):
        for j in range(phi_mesh.shape[1]):
            phi = int(phi_angles[j])
            psi = int(psi_angles[i])
            #print(phi,psi,phi_mesh[i,j],psi_mesh[i,j])
            kappa = 1000
            
            print(f'Laod data from constrained md phi={phi}°, theta={psi}°')
            colvar_name='COLVAR_restrained'
            data_path = f"ala2/simulation/constrained/phi_{phi}_psi_{psi}/"
            _,_,s_C7eq_xv_heavy,s_C7eq_fs_heavy = load_data(data_path,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
            
            colvar = read_COLVAR(data_path+colvar_name)
            
            u = torch.from_numpy(colvar[:,3]).to(device)
            ww = torch.exp(u/kbt).unsqueeze(1)
            ww = ww/torch.sum(ww)
            w_constrained = torch.cat((w_constrained,ww),dim=0)
            dihedral_phi = torch.from_numpy(colvar[:,1]).to(device)
            dihedral_psi = torch.from_numpy(colvar[:,2]).to(device)
            dphi = dihedral_phi-phi_mesh[i,j]
            dpsi = dihedral_psi-psi_mesh[i,j]
            dphi[dphi>np.pi] = dphi[dphi>np.pi]-2*np.pi
            dphi[dphi<-np.pi] = dphi[dphi<-np.pi]+2*np.pi
            dpsi[dpsi>np.pi] = dpsi[dpsi>np.pi]-2*np.pi
            dpsi[dpsi<-np.pi] = dpsi[dpsi<-np.pi]+2*np.pi
            cv_force_1 = kappa*torch.stack((dphi,dpsi),dim=-1)
            if torch.max(torch.abs(dphi))>np.pi:
                print(f"ERROR: phi{phi},psi{psi}")
            if torch.max(torch.abs(dpsi))>np.pi:
                print(f"ERROR: phi{phi},psi{psi}")
            #print(torch.max(torch.abs(u-kappa/2*(dphi**2+dpsi**2))))
            
            data_constrained = torch.cat((data_constrained,s_C7eq_xv_heavy.to(device)),dim=0)
            dU_constrained = torch.cat((dU_constrained,-s_C7eq_fs_heavy.to(device)),dim=0)
            cv_force = torch.cat((cv_force,cv_force_1),dim=0)
    data_constrained.requires_grad_(True)
    data_constrained_x = data_constrained[:,:xdim]
    
    y = cv_force*phipsi(data_constrained_x,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy)
    x_force = torch.autograd.grad(outputs=y, 
                                    inputs=data_constrained_x,
                                    grad_outputs=torch.ones_like(y),
                                    create_graph=False, retain_graph=False)[0]
    
    data_constrained.requires_grad_(False)
    dU_constrained = dU_constrained + x_force
    data_constrained = data_constrained.detach().to('cpu')
    dU_constrained = dU_constrained.detach().to('cpu')
    del x_force,cv_force,y
    torch.cuda.empty_cache()
'''
print('Test:',q(C7eq_xv_heavy)) 


d = q.descriptor(C7eq_xv_heavy)
d = d.detach().cpu().numpy()

dihedral_phi = np.arctan2(d[:, 1], d[:, 0]) * 180 / np.pi
dihedral_psi = np.arctan2(d[:, 3], d[:, 2]) * 180 / np.pi

bins = 300 # Number of bins along each dimension
hist, x_edges, y_edges = np.histogram2d(dihedral_phi, dihedral_psi, bins=bins, density=True)
epsilon = 1e-8
log_hist = np.log(hist + epsilon)
phi_grid, psi_grid = np.meshgrid(
    0.5 * (x_edges[:-1] + x_edges[1:]),  # Bin centers for phi
    0.5 * (y_edges[:-1] + y_edges[1:])   # Bin centers for psi
)

# Plot the 2D histogram
plt.figure(figsize=(8, 6))
contour = plt.contourf(phi_grid, psi_grid, log_hist.T, levels=20, cmap="viridis")
cbar = plt.colorbar(contour)
cbar.set_label("Log10 (Density)")

# Add labels and title
plt.xlabel("Phi (degrees)")
plt.ylabel("Psi (degrees)")
plt.title("2D Histogram of Phi and Psi Angles")
plt.grid(True)

# Show the plot
plt.show()
'''

part = 0.1
C7eq_xv_heavy = C7eq_xv_heavy[:int(C7eq_xv_heavy.shape[0]*part)]
C7ax_xv_heavy = C7ax_xv_heavy[:int(C7ax_xv_heavy.shape[0]*part)]
label_a = torch.zeros(C7eq_xv_heavy.shape[0], dtype=torch.float32)
label_b = torch.ones(C7ax_xv_heavy.shape[0], dtype=torch.float32)
data_boundary = torch.cat((C7eq_xv_heavy, C7ax_xv_heavy), dim=0)
label_boundary = torch.cat((label_a, label_b), dim=0).unsqueeze(1)

data_T = highT_xv_heavy
dU_T = -highT_fs_heavy 


data_long = torch.cat((long_C7eq_xv_heavy,long_C7ax_xv_heavy),dim = 0)
dU_long = -torch.cat((long_C7eq_fs_heavy,long_C7ax_fs_heavy),dim = 0)

normalized = False

if data_label =='highT':
    data = data_T
    dU = dU_T
elif data_label == 'long':
    data = data_long
    dU = dU_long

elif data_label == 'biased':
    data = data_biased
    dU = dU_biased
    w2 = torch.ones_like(w_biased)
    w2 = w2/torch.sum(w2)
    w = w2
elif data_label == 'biased_1':
    data = data_biased
    dU = dU_biased
    w2 = torch.ones_like(w_biased)
    w2 = w2/torch.sum(w2)
    w = w2
elif data_label == 'biased_2':
    data = torch.cat((data_metad,data_biased),dim = 0)
    dU = torch.cat((dU_metad,dU_biased),dim = 0)
    w1 = torch.ones((data_metad.shape[0],1),dtype=torch.float32,device=device)
    w1 = w1/torch.sum(w1)
    w2 = torch.ones_like(w_biased)
    w2 = w2/torch.sum(w2)
    w = torch.cat((w1,w2),dim=0)
elif data_label == 'all':
    data = torch.cat((data_T,data_biased),dim = 0)
    dU = torch.cat((dU_T,dU_biased),dim = 0)
    w1 = torch.ones((data_T.shape[0]+data_long.shape[0],1),dtype=torch.float32,device=device)
    w1 = w1/torch.sum(w1)
    w2 = torch.ones_like(w_biased)
    w2 = w2/torch.sum(w2)
    w = torch.cat((w1,w2),dim=0)
elif data_label == 'all_normalized':
    data = torch.cat((data_T,data_long,data_biased),dim = 0)
    dU = torch.cat((dU_T,dU_long,dU_biased),dim = 0)
    w1 = torch.ones((data_T.shape[0]+data_long.shape[0],1),dtype=torch.float32,device=device)
    w1 = w1/torch.sum(w1)
    w2 = torch.ones_like(w_biased)
    w2 = w2/torch.sum(w2)
    w = torch.cat((w1,w2),dim=0) 
    normalized=True
elif data_label == 'constrained':
    data = data_constrained
    dU = dU_constrained
    w = torch.ones_like(w_constrained)
    w = w/torch.sum(w)
elif data_label == 'highT_biased_1':
    data = data_T
    dU = dU_T
elif data_label == 'highT_biased_2':
    data = torch.cat((data_T,data_biased),dim = 0)
    dU = torch.cat((dU_T,dU_biased),dim = 0)
    w_T = torch.ones((data_T.shape[0],1), dtype=torch.float32, device=device)
    w_T = w_T/torch.sum(w_T)
    w_biased = torch.ones_like(w_biased)
    w_biased = w_biased/torch.sum(w_biased)
    w = torch.cat((w_T,w_biased),dim = 0)
elif data_label == 'metad_1':
    data = data_metad
    dU = dU_metad
    w = torch.ones((dU_metad.shape[0],1),device = device,dtype = torch.float32)
    w = w/torch.sum(w)

if use_distance:
    data_label = f"{data_label}_d45"


heavy_atom_mass = torch.ones_like(heavy_atom_mass).to(device)

data = data.to(device)
dU = dU.to(device)

if mask_AB:
    with torch.no_grad():
        c_C7eq = torch.tensor([[-1.46, 1.3305264]]).to(device)
        c_C7ax = torch.tensor([[1.01, -0.71]]).to(device)

        dihedrals = phipsi(data[:,:xdim], num_heavy_atoms, phi_group_heavy, psi_group_heavy)

        dC7eq = dihedrals - c_C7eq
        dC7ax = dihedrals - c_C7ax
        r = 10/180*np.pi  
        mask_eq = (dC7eq[:,0]**2 + dC7eq[:,1]**2) < r**2
        mask_ax = (dC7ax[:,0]**2 + dC7ax[:,1]**2) < r**2
        mask_CAB = ~(mask_eq |mask_ax)

        data = data[mask_CAB,:].detach()
        dU = dU[mask_CAB,:].detach()

logging.info(f'Data label: {data_label}')
logging.info(f'Data size: {data.shape[0]}')
print(f'Data size: {data.shape[0]}')

plt.figure()
phipsi_value = phipsi(data[:,:xdim],num_heavy_atoms,phi_group_heavy,theta_group_heavy)
plt.scatter(phipsi_value[:,0].detach().cpu().numpy(), phipsi_value[:,1].detach().cpu().numpy(),alpha = 0.1,s=1, c='black')
plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=10, cmap="viridis")
plt.savefig(f'ala2/fig/data_points.png')
plt.close()




total_loss_list = []
total_b_loss_list = []
total_pinn_loss_list = []
total_tot_loss_list = []
if use_distance:
    q = NNd2_45(layer_sizes=layers,n_atoms=num_heavy_atoms, activation='sigmoid')
else:
    q = NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy, activation='sigmoid')

'''
model_file = f'./ala2/model/gamma{gamma}_kbt{kbt}_{data_label}.pth'
config_file = f'./ala2/config/gamma{gamma}_kbt{kbt}_{data_label}.txt'

q = load_model(model_file,config_file)
'''
#q.load_state_dict(torch.load(f"ala2/model/gamma{gamma}_kbt{kbt}_{data_label}_subtrain_9.pth"))

#model_file = f'./model/gamma10_kbt0.5_1I.pth'
#config_file = f'./config/gamma10_kbt0.5_1I.txt'
#q = load_model(model_file,config_file)

logging.info(f'Muller potential with gamma={gamma}, kbt={kbt}')
logging.info(f'NN info: ')
logging.info(f'Layers: {layers},activation: {activ}')
logging.info(f'Number of samples: {data.shape[0]}')
logging.info(f'Using device: {device}')

    


# In[12]:


args['lam'] = 4
args['eta'] = 4

hyperparams_list = []
hyperparams_1 = {
    "lam": 4,
    "eta": 4,
    "batch_size": batch_size,
    "lr": 1e-4,
    "adaptive": False,
    "beta": 0.8,
    "alpha_beta": 0.9,
    "pinn_weight": 0,
    "grad_weight": 0.5,
    "NNt": Nt*2,          # assumes Nt is defined earlier
    "NNsteps": Nsteps * 1,  # assumes Nsteps is defined earlier
}

hyperparams_2 = {
    "batch_size": batch_size,
    # "eta": 4,   # commented out in original
    "lr": 1e-4,
    # "eta_alt": 1, # commented out in original
    # "lam": 1,     # commented out in original
    # "kbt": 0.5,   # commented out in original
    # increment subtrain_idx by 1 (store as delta)
    "NNsteps": Nsteps * 2,   # requires Nsteps defined
    "NNt": Nt,               # requires Nt defined
    "adaptive": False,
    "beta": 0.8,
    "alpha_beta": 0.95,
    "pinn_weight": 0,
    "grad_weight": 0.5,
    # parameters set into args
    "lam": 4,
    "eta": 4,
}

hyperparams_3 = {
    "batch_size": batch_size,
    # "eta": 4,  # commented out in original
    "lr": 1e-4,
    # increment subtrain_idx by 1 (store as delta so we can apply safely)
    "NNsteps": Nsteps * 2,   # requires Nsteps defined
    "NNt": Nt*2,               # requires Nt defined
    "adaptive": False,
    "beta": 0.8,
    "alpha_beta": 0.99,
    "pinn_weight": 0,
    "grad_weight": 0.5,
    # parameters to set in args
    "lam": 4,
    "eta": 4,
}

hyperparams_4 = {
    "batch_size": batch_size,
    # "eta": 4,  # commented out in original
    "lr": 1e-4,
    # increment subtrain_idx by 1 (store as delta so we can apply safely)
    "NNsteps": Nsteps * 2,   # requires Nsteps defined
    "NNt": Nt*2,               # requires Nt defined
    "adaptive": False,
    "beta": 0.8,
    "alpha_beta": 0.99,
    "pinn_weight": 0,
    "grad_weight": 0.5,
    # parameters to set in args
    "lam": 4,
    "eta": 4,
}

hyperparams_5 = {
    "batch_size": batch_size,
    # "eta": 4,  # commented out in original
    "lr": 1e-4,
    # increment subtrain_idx by 1 (store as delta so we can apply safely)
    "NNsteps": Nsteps *2,   # requires Nsteps defined
    "NNt": Nt*2,               # requires Nt defined
    "adaptive": False,
    "beta": 0.8,
    "alpha_beta": 0.99,
    "pinn_weight": 0,
    "grad_weight": 0.5,
    # parameters to set in args
    "lam": 10,
    "eta": 10,
}
hyperparams_6 = {
    "batch_size": batch_size,
    # "eta": 4,  # commented out in original
    "lr": 1e-4,
    # increment subtrain_idx by 1 (store as delta so we can apply safely)
    "NNsteps": Nsteps *2,   # requires Nsteps defined
    "NNt": Nt*2,               # requires Nt defined
    "adaptive": False,
    "beta": 0.8,
    "alpha_beta": 0.99,
    "pinn_weight": 0,
    "grad_weight": 0.5,
    # parameters to set in args
    "lam": 10,
    "eta": 10,
}
hyperparams_list.append(hyperparams_1)
hyperparams_list.append(hyperparams_2)
hyperparams_list.append(hyperparams_3)
hyperparams_list.append(hyperparams_4)
hyperparams_list.append(hyperparams_5)
for i in range(5):
    hyperparams_list.append(hyperparams_6)


gammas = []
data_labels = []
# Set variables from dict (simple explicit assignments)

hyperparams=hyperparams_1
batch_size = hyperparams["batch_size"]

lr = hyperparams["lr"]
adaptive = hyperparams["adaptive"]
beta = hyperparams["beta"]
alpha_beta = hyperparams["alpha_beta"]
pinn_weight = hyperparams["pinn_weight"]
grad_weight = hyperparams["grad_weight"]
NNt = hyperparams["NNt"]
NNsteps = hyperparams["NNsteps"]
args['lam'] = hyperparams['lam']
args['eta'] = hyperparams['eta']

q.to(device)
data = data.to(device)
torch.cuda.empty_cache()

'''
loss_list,b_loss_list,tot_loss_list,pinn_loss_list=train_mass(model=q,
                                          data=data,
                                          mass = heavy_atom_mass,
                                          w=w,
                                          batchsize=batch_size,
                                          data_b=data_boundary,
                                          label_b=label_boundary,
                                          alpha_b=1000,
                                          lr = 1e-3,
                                          num_tsteps=NNt,
                                          num_epoches=NNsteps,
                                          device=device,
                                          args=args,
                                          xdim=ndim,
                                          vdim=ndim,
                                          dU=dU,
                                          checkpoint=10,
                                          adaptive=adaptive,
                                          beta=beta,
                                          alpha_beta = alpha_beta,
                                          pinn_weight = pinn_weight, 
                                          grad_weight = grad_weight,
                                          alpha_l2 = 1e-4,
                                          normalized=normalized)
'''
for hyperparams in hyperparams_list:
    subtrain_idx +=1
    batch_size = hyperparams["batch_size"]

    lr = hyperparams["lr"]
    adaptive = hyperparams["adaptive"]
    beta = hyperparams["beta"]
    alpha_beta = hyperparams["alpha_beta"]
    pinn_weight = hyperparams["pinn_weight"]
    grad_weight = hyperparams["grad_weight"]
    NNt = hyperparams["NNt"]
    NNsteps = hyperparams["NNsteps"]
    args['lam'] = hyperparams['lam']
    args['eta'] = hyperparams['eta']

    q.to(device)
    data = data.to(device)
    

#for i in range(xdim_reduce):
#    print(f'atom {i}: mean x {mmm[i].item()}, mean v {mmm[num_heavy_atoms+i].item()}, std x {sss[i].item()}, std v {sss[i+num_heavy_atoms].item()}')

# kbt = 1
    logging.info(f'Subtraining index: {subtrain_idx}')
    logging.info(f'Batch size: {batch_size}')
    logging.info(f'Learning rate: {lr}')
    logging.info(f'Number of training steps: {NNsteps}')
    logging.info(f'Number of time steps: {NNt}')
    logging.info(f'Args: {args}')

    loss_list,b_loss_list,tot_loss_list,pinn_loss_list=train_mass(model=q,
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
                                            dU=dU,
                                            checkpoint=10,
                                            adaptive=adaptive,
                                            beta=beta,
                                            alpha_beta = alpha_beta,
                                            pinn_weight = pinn_weight, 
                                            grad_weight = grad_weight,
                                            alpha_l2 = 1e-3,
                                            normalized=normalized,
                                            resampling_num=10)
    total_loss_list += loss_list
    total_b_loss_list += b_loss_list
    total_pinn_loss_list += pinn_loss_list
    total_tot_loss_list += tot_loss_list
    # In[14]:
    fig_file = f'ala2/fig/loss_gamma{gamma}_kbt{kbt}_{data_label}_subtrain_{subtrain_idx}.png'
    plot_loss(loss_list,b_loss_list,tot_loss_list,pinn_loss_list,fig_file)

    model_file = f'./ala2/model/gamma{gamma}_kbt{kbt}_{data_label}_subtrain_{subtrain_idx}.pth'
    config_file = f'./ala2/config/gamma{gamma}_kbt{kbt}_{data_label}_subtrain_{subtrain_idx}.txt'
    save_model(q,model_file,config_file)
    gammas.append(gamma)
    data_labels.append(data_label+f'_subtrain_{subtrain_idx}')




# Adjust layout  
fig_file = f'ala2/fig/loss_gamma{gamma}_kbt{kbt}_{data_label}_subtrain_{subtrain_idx}.png'
plot_loss(loss_list,b_loss_list,tot_loss_list,pinn_loss_list,fig_file)


model_file = f'./ala2/model/gamma{gamma}_kbt{kbt}_{data_label}_subtrain_{subtrain_idx}.pth'
config_file = f'./ala2/config/gamma{gamma}_kbt{kbt}_{data_label}_subtrain_{subtrain_idx}.txt'
save_model(q,model_file,config_file)


fig_file = f'ala2/fig/loss_gamma{gamma}_kbt{kbt}_{data_label}_total.png'
plot_loss(total_loss_list,total_b_loss_list,total_tot_loss_list,total_pinn_loss_list,fig_file)

# Adjust layout  
model_file = f'./ala2/model/gamma{gamma}_kbt{kbt}_{data_label}.pth'
config_file = f'./ala2/config/gamma{gamma}_kbt{kbt}_{data_label}.txt'
save_model(q,model_file,config_file)


quick_diagnose(gammas,data_labels,layers,use_distance=use_distance)
#layers_0 = [xdim_reduce,8, 64, 64,8, 1]
#distilling_models_phipsi(gammas,data_labels,layers,layers_0)



