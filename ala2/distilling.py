#!/usr/bin/env python
# coding: utf-8


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

import copy
import matplotlib.pyplot as plt
from model_training import train_resample,pinn_loss,build_rightside, train_mass,train_overdamped
from hist import hist_reweight
from utils import *

import logging

# Configure logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
topology_file = "ala2/simulation/topol.top"
mass = read_mass(topology_file)
mass=torch.tensor(mass).to(device)
heavy_atom_mass = mass[heavy_atom_indices]
mass = torch.repeat_interleave(mass,3).unsqueeze(0)
heavy_atom_mass = torch.repeat_interleave(heavy_atom_mass,3).unsqueeze(0)
num_heavy_atoms = heavy_atom_indices.shape[0]
kbt = 300 * 0.0083144621

def generate_mean_q(qs,step,k):
    deg_to_rad = np.pi / 180.0
    deg_to_rad = np.float32(deg_to_rad)
    phi_angles = np.arange(-180, 180, step).astype(np.float32)
    psi_angles = np.arange(-180, 180, step).astype(np.float32)
    phi_mesh, psi_mesh = np.meshgrid(phi_angles, psi_angles)
    phi_mesh = torch.from_numpy(phi_mesh * deg_to_rad).to(device)
    psi_mesh = torch.from_numpy(psi_mesh * deg_to_rad).to(device)
    q_values = torch.zeros((len(qs),phi_mesh.shape[0],phi_mesh.shape[1]),dtype=torch.float32,device=device)

    with torch.no_grad():
        for i in range(phi_mesh.shape[0]):
            for j in range(phi_mesh.shape[1]):
                phi = int(phi_angles[j])
                psi = int(psi_angles[i])
                #print(phi,psi,phi_mesh[i,j],psi_mesh[i,j])
                
                print(f'Generating data for phi={phi}°, theta={psi}°')
                xs = read_xvg(f"ala2/simulation/constrained/phi_{phi}_psi_{psi}/positions.xvg")
                xs = torch.from_numpy(xs.astype(np.float32)).to(device)
                xs = xs[:, heavy_dim_indices]

                
                repeated_xs = xs.repeat(k, 1)  # Shape: (k*m, n)

                # Step 2: Sample from normal distribution for the last n dimensions
                # Each column uses a different sigma
                sigmas = kbt/torch.sqrt(heavy_atom_mass)  # Shape: (1, n)
                #print(xs.shape,sigmas.shape)
                noise = torch.randn(k * xs.shape[0], xs.shape[1],device = device) * sigmas  # Shape: (k*m, n)

                # Step 3: Concatenate repeated_xs and noise along the second dimension
                data_phi_psi = torch.cat([repeated_xs, noise], dim=1)  # Shape: (k*m, 2n)
                data_phi_psi = data_phi_psi.to(device)
                for idx_q in range(len(qs)):
                    q = qs[idx_q]
                    q_values[idx_q,i,j] = torch.mean(q(data_phi_psi)).detach()
                del data_phi_psi
                torch.cuda.empty_cache()
    return phi_mesh, psi_mesh, q_values

def distilling_phipsi(q0:NNphipsi_overdamped,phipsi,data_q,num_epochs,print_every=10,lr = 1e-3,batch_size=64):

    optimizer = optim.Adam(q0.parameters(), lr=lr)
    # Optional: scheduler
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # ======== 5) Training loop =========




    for epoch in range( 1, num_epochs + 1):
        train_loss = 0.0
        loss = (q0.d_forward(phipsi)-data_q)**2
        train_loss = loss.mean().item()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.8f} ")

        # save best
        
def distilling_models_phipsi(gammas,data_labels,layers,layers_0,
                             n_epochs = 4000,
                             print_every = 100,
                             k=10000,
                             step = 10):

    num_heavy_atoms = heavy_atom_indices.shape[0]


    #xdim_reduce = 45


      # kBT in kcal/mol   

    batch_size = 2048 #not implement


    activ  = 'sigmoid'
    

    qs = []
    q0s = []

    model_files=[]
    config_files=[]

    for gamma, data_label in zip(gammas, data_labels):
        qs.append(NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy, activation=activ))
        q0s.append(NNphipsi_overdamped(layer_sizes=layers_0,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy, activation=activ))
        qs[-1].load_state_dict(torch.load(f"ala2/model/gamma{gamma}_kbt{kbt}_{data_label}.pth"))
        qs[-1].to(device)
        q0s[-1].to(device)
        distilling_path=f"ala2/model/distilling_gamma{gamma}"
        distilling_config_path=f"ala2/config/distilling_gamma{gamma}"
        if not os.path.exists(distilling_path):
            os.makedirs(distilling_path)
        if not os.path.exists(distilling_config_path):
            os.makedirs(distilling_config_path)
        model_files.append(f"{distilling_path}/gamma{gamma}_{data_label}.pth")
        config_files.append(f"{distilling_config_path}/gamma{gamma}_{data_label}.txt")



    phi_mesh, theta_mesh, q_values = generate_mean_q(qs,step,k=k)
    phi_mesh = phi_mesh.flatten()
    theta_mesh = theta_mesh.flatten()
    phi = phi_mesh.detach().cpu().numpy()
    theta = theta_mesh.detach().cpu().numpy()

    for i in range(len(qs)):
        q_values_i = q_values[i,:,:].flatten().unsqueeze(1)
        q0 = q0s[i]
        config_file = config_files[i]
        model_file = model_files[i]
        data = torch.stack((torch.sin(phi_mesh),torch.cos(phi_mesh), torch.sin(theta_mesh), torch.cos(theta_mesh)), dim=1)
        data = data.detach()
        distilling_phipsi(q0,data,q_values_i,num_epochs=n_epochs,print_every=print_every,lr=1e-2,batch_size=batch_size)
        distilling_phipsi(q0,data,q_values_i,num_epochs=n_epochs,print_every=print_every,lr=1e-3,batch_size=batch_size)
        distilling_phipsi(q0,data,q_values_i,num_epochs=n_epochs,print_every=print_every,lr=1e-4,batch_size=batch_size)
        

        
        
        q0.save(data,config_file,model_file,is_description=True)


if __name__ == "__main__":
    gamma_data_label_file = "ala2/distilling_gamma_data_label.txt"
    gammas, data_labels = get_gamma_data_label(gamma_data_label_file)
    num_heavy_atoms = heavy_atom_indices.shape[0]
    xdim = heavy_atom_indices.shape[0] * 3
    xdim_reduce = 4
    vdim_reduce = 4

    #xdim_reduce = 45

    ndim = xdim
    layers = [xdim_reduce+vdim_reduce,8,64,64,64,64,8,1]
    layers_0 = [xdim_reduce,8, 64, 64,8, 1]
    distilling_models_phipsi(gammas,data_labels,layers,layers_0,step=10)
