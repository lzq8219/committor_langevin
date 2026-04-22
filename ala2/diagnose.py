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
from fes import U_phipsi,U_phitheta,phi_contour_1,psi_contour_1,phi_contour_2,theta_contour_2

epsilon = 1e-10

gamma_o = 10040
kbt = 300 * 0.0083144621  # kBT in kcal/mol 
num_heavy_atoms = heavy_atom_indices.shape[0]
xdim = heavy_atom_indices.shape[0] * 3
vdim = heavy_atom_indices.shape[0] * 3
xdim_reduce = 4
vdim_reduce = 4



#data_label = 'long'
#data_label = 'long_1'
#data_label = 'long_2'
#data_label = 'long_3'
#data_label = 'biased'
#data_label = 'all'

#xdim_reduce = 45
#vdim_reduce = 45  

activ  = 'sigmoid'
device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

#mass info
topology_file = "ala2/simulation/topol.top"
mass = read_mass(topology_file)
mass=torch.tensor(mass)
heavy_atom_mass = mass[heavy_atom_indices]
mass = torch.repeat_interleave(mass,3).unsqueeze(0)
heavy_atom_mass = torch.repeat_interleave(heavy_atom_mass,3).unsqueeze(0)
num_heavy_atoms = heavy_atom_indices.shape[0]

# data loading





'''
if data_label =='highT':
    data = data_T
    dU = dU_T
elif data_label == 'long':
    data = data_long
    dU = dU_long
elif data_label == 'long_1':
    data = data_long_1
    dU = dU_long_1
elif data_label == 'long_2':
    data = data_long_2
    dU = dU_long_2
elif data_label == 'long_3':
    data = data_long_3
    dU = dU_long_3
elif data_label == 'biased':
    data = torch.cat((data_long_1,data_long_2,data_long_3),dim = 0)
    dU = torch.cat((dU_long_1,dU_long_2,dU_long_3),dim = 0)
elif data_label == 'all':
    data = torch.cat((data_T,data_long,data_long_1,data_long_2,data_long_3),dim = 0)
    dU = torch.cat((dU_T,dU_long,dU_long_1,dU_long_2,dU_long_3),dim = 0)
'''







# q = NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=psi_group_heavy, activation='sigmoid')
#q = NNd2_45(layer_sizes=layers,n_atoms=num_heavy_atoms, activation='sigmoid')
#q = NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy, activation='sigmoid')
#q.load_state_dict(torch.load(f"ala2/model/gamma{gamma}_kbt{kbt}_{data_label}.pth"))
#model_path_o = f"ala2/model/distilling_gamma0.20005/gamma0.20005_highT_subtrain_2.pth"
#config_path_o = f"ala2/config/distilling_gamma0.20005/gamma0.20005_highT_subtrain_2.txt"
#q0 = load_model_phipsi(model_path_o,config_path_o)


#q0.to(device)

def draw_q0_dq0(q0,n,data,figname_q0,filename_dq0,device=device,q_old = None):
    # Define the number of points along each axis
    q0 = q0.to(device)
    n = 360
    linspace = torch.linspace(-np.pi, np.pi, n)
        
        # Create a meshgrid from the 1D tensor
    grid_x, grid_y = torch.meshgrid(linspace, linspace, indexing='ij')

    

    # Stack the grids along the last dimension and reshape to (n*n, 2)
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)


    grid = grid.to(device)


    xxx = torch.cat((torch.sin(grid[:,0:1]),torch.cos(grid[:,0:1]),torch.sin(grid[:,1:2]),torch.cos(grid[:,1:2])),dim=1)
    print(xxx.shape)
    xxx.requires_grad_(True)
    if q_old is not None:
        q0_value = q0.d_forward(xxx) - q_old.d_forward(xxx)
    else:
        q0_value = q0.d_forward(xxx)

    dq0 = torch.autograd.grad(q0_value, xxx, torch.ones_like(q0_value), create_graph=True)[0]
    dq0_norm2 = torch.sum(dq0**2, dim=1, keepdim=True)
    dq0_norm2 = torch.log(dq0_norm2+1e-10)
    dq0_norm2 = dq0_norm2.detach().cpu().numpy()
    dq0_norm2 = dq0_norm2.reshape(n,n)


    q0_value = q0_value.detach().cpu().numpy()
    q0_value = q0_value.reshape(n,n)




    grid_value = grid.detach().cpu().numpy()

    print(data.shape)
    phi_mesh = grid_x.detach().cpu().numpy()
    psi_mesh = grid_y.detach().cpu().numpy()
    plt.figure(figsize=(8,6))
    #plt.scatter(grid_value[:,0], grid_value[:,1], c=q0_value,  cmap='turbo')
    plt.contourf(phi_mesh,psi_mesh, q0_value, cmap='turbo',levels=10)
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])
    plt.colorbar(label='$q_0$')  # Add a colorbar to show the range of mean values
    contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=10, cmap="viridis")  # levels 控制等高线的数量
    plt.clabel(contour_lines, inline=True, fontsize=8)
    phipsi_value = phipsi(data,num_heavy_atoms,phi_group_heavy,theta_group_heavy)
    plt.scatter(phipsi_value[:,0].detach().cpu().numpy(), phipsi_value[:,1].detach().cpu().numpy(),alpha = 0.1,s=1, c='black')

    plt.xlabel('$\phi$')
    plt.ylabel('$ \\theta$')
    plt.title('committor function ')
    plt.savefig(figname_q0, dpi=300)
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.contourf(phi_mesh,psi_mesh, dq0_norm2, cmap='turbo',levels=10)
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])
    plt.colorbar(label='d$q_0$')  # Add a colorbar to show the range of mean values
    contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=10, cmap="viridis")  # levels 控制等高线的数量
    plt.clabel(contour_lines, inline=True, fontsize=8)
    plt.scatter(phipsi_value[:,0].detach().cpu().numpy(), phipsi_value[:,1].detach().cpu().numpy(),alpha = 0.1,s=1, c='black')
    plt.xlabel('$\phi$')
    plt.ylabel('$ \\theta$')
    plt.title('committor function ')
    
    plt.savefig(filename_dq0, dpi=300)
    plt.clf()

    torch.cuda.empty_cache()

def draw_q(qs,gammas,data_labels,data,step,num_v,mmm,nnn,vs,k,figname_mqs,figname_q_vslices,figname_q_datas,figname_mq_datas,figname_data_logps):
    num_models = len(qs)
    deg_to_rad = np.pi / 180.0
    deg_to_rad = np.float32(deg_to_rad)
    phi_angles = np.arange(-180, 180+step, step).astype(np.float32)
    psi_angles = np.arange(-180, 180+step, step).astype(np.float32)
    phi_mesh, psi_mesh = np.meshgrid(phi_angles, psi_angles)
    phi_mesh = torch.from_numpy(phi_mesh * deg_to_rad).to(device)
    psi_mesh = torch.from_numpy(psi_mesh * deg_to_rad).to(device)
    q_values = torch.zeros((num_models,phi_mesh.shape[0], phi_mesh.shape[1])).to(device)
    q_values_at_v = torch.zeros((num_models,num_v, phi_mesh.shape[0], phi_mesh.shape[1])).to(device)

    with torch.no_grad():
        for i in range(phi_mesh.shape[0]):
            for j in range(phi_mesh.shape[1]):
                phi = int(phi_angles[j])
                psi = int(psi_angles[i])
                if phi == 180 or psi == 180:
                    continue
                print(f'Generating data for phi={phi}°, theta={psi}°')
                xs = read_xvg(f"ala2/simulation/constrained/phi_{phi}_psi_{psi}/positions.xvg")
                xs = torch.from_numpy(xs.astype(np.float32)).to(device)
                xs = xs[:, heavy_dim_indices]
                for iii in range(mmm):
                    for jjj in range(nnn):
                        idx = iii*nnn+jjj
                        v = vs[idx:idx+1,:]
                        v = v.repeat(xs.shape[0],1)
                        data_phi_psi = torch.cat((xs,v),dim=1)
                        data_phi_psi = data_phi_psi.to(device)
                        
                        with torch.no_grad():
                            for iiii in range(num_models):
                                q = qs[iiii]
                                q_values_at_v[iiii,idx,i,j] = torch.mean(q(data_phi_psi)).detach()
                        del data_phi_psi
                        torch.cuda.empty_cache()

                
                repeated_xs = xs.repeat(k, 1)  # Shape: (k*m, n)

                # Step 2: Sample from normal distribution for the last n dimensions
                # Each column uses a different sigma
                sigmas = kbt/torch.sqrt(heavy_atom_mass).to(device)  # Shape: (1, n)
                #print(xs.shape,sigmas.shape)
                noise = torch.randn(k * xs.shape[0], xs.shape[1],device = device) * sigmas  # Shape: (k*m, n)

                # Step 3: Concatenate repeated_xs and noise along the second dimension
                data_phi_psi = torch.cat([repeated_xs, noise], dim=1)  # Shape: (k*m, 2n)
                data_phi_psi = data_phi_psi.to(device)
                for iiii in range(num_models):
                    q = qs[iiii]
                    q_values[iiii,i,j] = torch.mean(q(data_phi_psi)).detach()
                del data_phi_psi
                torch.cuda.empty_cache()
                #print(torch.std(q(data_phi_psi)))

                #print(phi, q.descriptor(data_phi_psi)[:,0].mean().item(),q.descriptor(data_phi_psi)[:,1].mean().item())
                #print(psi, q.descriptor(data_phi_psi)[:,2].mean().item(),q.descriptor(data_phi_psi)[:,3].mean().item())
                #print(i,j,q_value)

    q_values[:,:,-1] = q_values[:,:,0]
    q_values[:,-1,:] = q_values[:,0,:]
    q_values_at_v[:,:,-1,:] = q_values_at_v[:,:,0,:]
    q_values_at_v[:,-1,:,:] = q_values_at_v[:,0,:,:]

    phi_mesh, psi_mesh = np.meshgrid(phi_angles*deg_to_rad, psi_angles*deg_to_rad)
    q_values = q_values.detach().cpu().numpy()
    for iiii in range(num_models):
        q_values_i = q_values[iiii,:,:]
        plt.figure(figsize=(8,6))
        plt.contourf(phi_mesh,psi_mesh, q_values_i, cmap='turbo',levels=10)
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])
        plt.tight_layout()
        plt.colorbar(label='$q$')  # Add a colorbar to show the range of mean values
        contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=10, cmap="viridis")  # levels 控制等高线的数量
        plt.clabel(contour_lines, inline=True, fontsize=8)
        plt.xlabel('$\phi$ (degrees)')
        plt.ylabel('$\\theta$ (degrees)')
        plt.title('Mean Committor Function')
        plt.savefig(figname_mqs[iiii], dpi=300)
        plt.clf()
        plt.close()

    for iiii in range(num_models):
        fig, axes = plt.subplots(mmm, nnn, figsize=(8*mmm, 6*nnn))
        im = None
        for iii in range(mmm):
            for jjj in range(nnn):
                idx = iii*nnn+jjj
                ax = axes[iii, jjj]
                q_values_at_v_i = q_values_at_v[iiii,idx,:,:].detach().cpu().numpy()
                
                im = ax.contourf(phi_mesh,psi_mesh, q_values_at_v_i, cmap='turbo',levels=10)
                contour_lines = ax.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=10, cmap="viridis")  # levels 控制等高线的数量
                ax.clabel(contour_lines, inline=True, fontsize=8)
                ax.set_xlim([-np.pi, np.pi])
                ax.set_ylim([-np.pi, np.pi])
                
                ax.set_xlabel('$\phi$')
                ax.set_ylabel('$\\theta$')
                #ax.set_title(f'Mean Committor Function at v {idx}')
                
                del q_values_at_v_i
        #plt.tight_layout()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('committor function $q$')
    

        fig.savefig(figname_q_vslices[iiii], dpi=300)
        plt.clf()
        plt.close()

        torch.cuda.empty_cache()

    '''
    for g in grid:
        phi_deg, psi_deg, _, _ = g
        dir_name = f"phi_{phi_deg}_psi_{psi_deg}"
        plot_colvar(
            f"{dir_name}/COLVAR_restrained",
            f"{dir_name}/COLVAR_restrained.png")
    '''

    ## need to fix for multiple models
    '''
    data = data.to(device)

    q_value = q(data)
    phipsi_value = phipsi(data[:,:xdim],num_heavy_atoms,phi_group_heavy,theta_group_heavy)
    plt.scatter(phipsi_value[:,0].detach().cpu().numpy(), phipsi_value[:,1].detach().cpu().numpy(), c=q_value.detach().cpu().numpy(), cmap='viridis', s=1)
    plt.colorbar(label='$q$')  # Add a colorbar to show the range of mean values
    contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=10, cmap="viridis")  # levels 控制等高线的数量
    plt.clabel(contour_lines, inline=True, fontsize=8)
    plt.xlabel('$\phi$')
    plt.ylabel('$\\theta$')
    plt.savefig(figname_mq_data, dpi=300)
    plt.clf()

    q_value = q_value.detach().cpu().numpy()+1
    phipsi_value = phipsi_value.detach().cpu().numpy()
    args_hist = {}
    args_hist['xmin'] = phipsi_value[:,0].min()
    args_hist['xmax'] = phipsi_value[:,0].max()
    args_hist['ymin'] = phipsi_value[:,1].min()
    args_hist['ymax'] = phipsi_value[:,1].max()
    args_hist['xbins'] = 100
    args_hist['ybins'] = 100

    hist2d_m,xedges,yedges = hist2d_mean(phipsi_value[:,0], phipsi_value[:,1], q_value,args=args_hist)

    H, xedges, yedges = np.histogram2d(phipsi_value[:,0], phipsi_value[:,1], bins=[args_hist['xbins'], args_hist['ybins']], range=[[args_hist['xmin'], args_hist['xmax']], [args_hist['ymin'], args_hist['ymax']]])


    plt.figure(figsize=(6,5))
    contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=10, cmap="viridis")  # levels 控制等高线的数量
    plt.clabel(contour_lines, inline=True, fontsize=8)
    plt.pcolormesh(xedges, yedges, hist2d_m, shading='auto', cmap='turbo')
    plt.colorbar(label='Mean Value')  # Add a colorbar to show the range of mean values
    plt.xlabel('$\phi$')
    plt.ylabel('$ \\theta$')
    plt.title('committor function ')
    plt.savefig(figname_q_data, dpi=300)
    plt.clf()

    plt.figure(figsize=(6,5))
    contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=10, cmap="viridis")  # levels 控制等高线的数量
    plt.clabel(contour_lines, inline=True, fontsize=8)

    plt.pcolormesh(xedges, yedges, np.log(H.T+epsilon), shading='auto', cmap='turbo')
    plt.colorbar(label='Mean Value')  # Add a colorbar to show the range of mean values
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Histogram with Mean Values')
    plt.savefig(figname_data_logp, dpi=300)
    plt.clf()
    torch.cuda.empty_cache()
    '''







def quick_diagnose(gammas,data_labels,layers,
    step = 10,
    num_v = 4,
    mmm = 2,
    nnn = 2,
    k = 10000):
    qs = []
    figname_mqs = []
    figname_q_datas = []
    figname_mq_datas = []
    figname_data_logps = []
    figname_q_vslices = []
    filename_vs = f"ala2/model/vs_{kbt}.txt"
    vs = np.loadtxt(filename_vs)
    vs = torch.from_numpy(vs.astype(np.float32)).to(device)
    for gamma, data_label in zip(gammas, data_labels):
        qs.append(NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy, activation='sigmoid'))
        qs[-1].load_state_dict(torch.load(f"ala2/model/gamma{gamma}_kbt{kbt}_{data_label}.pth"))
        qs[-1].to(device)
        if not os.path.exists(f"ala2/fig/gamma{gamma}"):
            os.makedirs(f"ala2/fig/gamma{gamma}")

        figname_mqs.append(f'ala2/fig/gamma{gamma}/mean_q_gamma{gamma}_{data_label}.png')
        figname_q_datas.append(f'ala2/fig/gamma{gamma}/q_data_gamma{gamma}_{data_label}.png')
        figname_mq_datas.append(f'ala2/fig/gamma{gamma}/mq_data_gamma{gamma}_{data_label}.png')
        figname_data_logps.append(f'ala2/fig/gamma{gamma}/data_logp_gamma{gamma}_{data_label}.png')
        figname_q_vslices.append(f'ala2/fig/gamma{gamma}/vslice_committor_gamma{gamma}_{data_label}.png')

    data_all = []
    draw_q(qs,gammas,data_labels,data_all,step,num_v,mmm,nnn,vs,k,figname_mqs,figname_q_vslices,figname_q_datas,figname_mq_datas,figname_data_logps)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    layers = [xdim_reduce+vdim_reduce,8,64,64,64,64,8,1]
    layers_0 = [xdim_reduce, 64, 64, 1]
    gamma_data_label_file = "ala2/gamma_data_label.txt"
    gammas, data_labels = get_gamma_data_label(gamma_data_label_file)

    for i in range(10):
        layers_0 = [xdim_reduce,8, 64, 64,8, 1]
        figname_q0 = f'ala2/fig/q0_{i}.png'
        filename_dq0 = f'ala2/fig/dq0_{i}.png'
        model_file = f'ala2/model/o_gamma{10100+i}_kbt2.4943386299999997.pth'
        config_file = f'ala2/config/o_gamma{10100+i}_kbt2.4943386299999997.txt'
        q0 = load_model_phipsi(model_file,config_file,layers_0)
        data_l_eq = read_xvg(f"ala2/simulation/o_{101+i}/long_C7eq_-3/positions.xvg")
        data_l_ax = read_xvg(f"ala2/simulation/o_{101+i}/long_C7ax_-3/positions.xvg")
        data_l = torch.from_numpy(np.concatenate((data_l_eq,data_l_ax),axis=0).astype(np.float32)).to(device)
        data_l = data_l[:,heavy_dim_indices]
        draw_q0_dq0(q0,360,data_l,figname_q0,filename_dq0,device=device)




#draw_q0_dq0(q0,360,data_biased,figname_q0,filename_dq0,device=device)
    # quick_diagnose(gammas,data_labels,layers,step = 10)

    '''
    highT_path = "ala2/simulation/1500K/"  # Working directory for intermediate files
    C7eq_path = "ala2/simulation/long_C7eq/"
    C7ax_path = "ala2/simulation/long_C7ax/"
    long_C7eq_path = "ala2/simulation/long_C7eq/"
    long_C7ax_path = "ala2/simulation/long_C7ax/"
    long_C7eq_path_1 = "ala2/simulation/long_C7eq_1/"
    long_C7ax_path_1 = "ala2/simulation/long_C7ax_1/"
    long_C7eq_path_2 = "ala2/simulation/long_C7eq_2/"
    long_C7ax_path_2 = "ala2/simulation/long_C7ax_2/"
    long_C7eq_path_3 = "ala2/simulation/long_C7eq_333/"
    long_C7ax_path_3 = "ala2/simulation/long_C7ax_333/"
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

    data_biased = np.loadtxt(f"ala2/simulation/data_gamma_10030kbt_2.4943386299999997.txt")
    data_biased = torch.from_numpy(data_biased.astype(np.float32))

    data_all = torch.cat((C7eq_xv_heavy[:,:xdim],C7ax_xv_heavy[:,:xdim],highT_xv_heavy[:,:xdim],data_biased),dim = 0).to(device)
    v = torch.normal(0, 1, size=data_all.shape, device=device)*np.sqrt(kbt)/torch.sqrt(heavy_atom_mass).to(device)
    data_all = torch.cat((data_all,v),dim=1)


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
    dU_long = -torch.cat((long_C7eq_fs_heavy,long_C7ax_fs_heavy),dim = 0
    '''
    