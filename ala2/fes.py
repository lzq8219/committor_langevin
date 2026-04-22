import sys
import os
origin_directory = os.getcwd()
model_directory = os.path.join(origin_directory, 'ala2')
src_directory = os.path.join(origin_directory, 'src')
sys.path.append(src_directory)
sys.path.append(model_directory)

import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import *

kbt = 2.4943386299999997  # kJ/mol
colvar = np.loadtxt('ala2/simulation/metad/COLAR_METAD_reweight')
phi = colvar[:,1]
psi = colvar[:,2]
theta = colvar[:,3]
weight = np.exp((colvar[:,4]-np.min(colvar[:,4]))/kbt)


args_hist = dict()
args_hist['xmin'] = -np.pi
args_hist['xmax'] = np.pi
args_hist['ymin'] = -np.pi
args_hist['ymax'] = np.pi
args_hist['xbins'] = 180
args_hist['ybins'] = 180
hist_phitheta,phi_contour_2,theta_contour_2 = hist2d_mean(phi, theta, weight, args=args_hist,mean=False)
hist_phitheta = hist_phitheta/np.sum(hist_phitheta)
phi_contour_2,theta_contour_2 = np.meshgrid(phi_contour_2[:-1], theta_contour_2[:-1])
hist_phipsi,phi_contour_1,psi_contour_1 = hist2d_mean(phi, psi, weight, args=args_hist,mean=False)
hist_phipsi = hist_phipsi/np.sum(hist_phipsi)
phi_contour_1,psi_contour_1 = np.meshgrid(phi_contour_1[:-1], psi_contour_1[:-1])



U_phipsi = -kbt * np.log(hist_phipsi + 1e-10)
U_phipsi = U_phipsi - np.min(U_phipsi)
#U_phipsi = gaussian_filter(U_phipsi, sigma=1)
U_phitheta = -kbt * np.log(hist_phitheta + 1e-10)
U_phitheta = U_phitheta - np.min(U_phitheta)
#U_phitheta = gaussian_filter(U_phitheta, sigma=1)

if __name__ == "__main__":
    data_paths = []
    #data_paths = [f"ala2/simulation/long_C7eq/positions.xvg",
    #              f"ala2/simulation/long_C7ax/positions.xvg"]
    for k in range(1,10):
        data_paths = []
        #for i in range(k,k+1):
        data_paths += [f"ala2/simulation/o_{k}/long_C7eq/positions.xvg",
                    f"ala2/simulation/o_{k}/long_C7ax/positions.xvg"]
        data = []

        for data_path in data_paths:
            data.append(np.loadtxt(data_path, comments=("#", "@"))[:,1:])
        data = np.concatenate(data, axis=0)
        data = data.reshape(data.shape[0],data.shape[1]//3,3)
        data = torch.from_numpy(data).float()
        #print(data.shape,data)
        _,_,phi = compute_dihedral_cossin(data[:,phi_group,:])
        _,_,theta = compute_dihedral_cossin(data[:,theta_group,:])
        _,_,psi = compute_dihedral_cossin(data[:,psi_group,:])
        #print(phi.shape,psi.shape)
        phi = phi.numpy()
        psi = psi.numpy()
        theta = theta.numpy()
        weight = torch.ones(phi.shape[0]).numpy()
        #hist_phitheta,phi_contour_2,theta_contour_2 = hist2d_mean(phi, theta, weight, args=args_hist,mean=False)
        #hist_phitheta = hist_phitheta/np.sum(hist_phitheta)
        #phi_contour_2,theta_contour_2 = np.meshgrid(phi_contour_2[:-1], theta_contour_2[:-1])
        


        U_phipsi = -kbt * np.log(hist_phipsi + 1e-10)
        U_phipsi = U_phipsi - np.min(U_phipsi)
        #U_phipsi = gaussian_filter(U_phipsi, sigma=1)
        U_phitheta = -kbt * np.log(hist_phitheta + 1e-10)
        U_phitheta = U_phitheta - np.min(U_phitheta)

        plt.figure(figsize=(6, 5))
        plt.scatter(phi, theta, s=1, alpha=0.1)
        plt.savefig('ala2/fig/temp1.png')


        plt.figure(figsize=(6, 5))
        contour_lines = plt.contour(phi_contour_1, psi_contour_1, U_phipsi, levels=10, cmap="viridis")  # levels 控制等高线的数量
        plt.clabel(contour_lines, inline=True, fontsize=8)  
        plt.colorbar(label='Free Energy (kJ/mol)')  # Add a colorbar to show the range of mean values
        plt.savefig('ala2/fig/temp2.png')

        plt.figure(figsize=(6, 5))
        plt.scatter(phi, theta, s=1, alpha=0.1)
        contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=10, cmap="viridis")  # levels 控制等高线的数量
        plt.clabel(contour_lines, inline=True, fontsize=8)
        plt.colorbar(label='Free Energy (kJ/mol)')  # Add a colorbar to show the range of mean values
        
        plt.savefig(f'ala2/fig/temp3_{k}.png')
        plt.close()
        print('Done!')
