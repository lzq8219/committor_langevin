import sys
import os
origin_directory = os.getcwd()
model_directory = os.path.join(origin_directory, 'ala2')
src_directory = os.path.join(origin_directory, 'src')
sys.path.append(src_directory)
sys.path.append(model_directory)
import torch
import numpy as np

import matplotlib.pyplot as plt

from utils import *

from gro import parsed,write_gro_from_torch
import time
from fes import U_phipsi,U_phitheta,phi_contour_1,psi_contour_1,phi_contour_2,theta_contour_2


epsilon = 1e-10

gamma_o = 10040
kbt = 300 * 0.0083144621  # kBT in kcal/mol 
num_heavy_atoms = heavy_atom_indices.shape[0]
xdim = heavy_atom_indices.shape[0] * 3
vdim = heavy_atom_indices.shape[0] * 3
xdim_reduce = 4
vdim_reduce = 4

layers = [xdim_reduce+vdim_reduce,8,256,256,1]
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


heavy_atom_mass = heavy_atom_mass.to(device)
resample = True

if __name__ = '__main__':

    gamma_data_label_file = "ala2/find_iso_gamma_data_label.txt"
    gammas, data_labels = get_gamma_data_label(gamma_data_label_file)
    qs = []
    for gamma, data_label in zip(gammas, data_labels):
        positions_filename = "positions.xvg"
        velocities_filename = "velocities.xvg"
        forces_filename = "forces.xvg"

        qs.append(NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy, activation='sigmoid'))
        qs[-1].load_state_dict(torch.load(f"ala2/model/gamma{gamma}_kbt{kbt}_{data_label}.pth",map_location=torch.device('cpu')))
        qs[-1].to(device)

    data_biased = torch.tensor([]).to(device)
    
    data_biased_xv = torch.tensor([]).to(device)
    
    itrs=10
    for i in range(1,itrs+1):

        C7eq_path_biased = f"ala2/simulation/o_{100+i}/long_C7eq_{-3.5}/"
        C7ax_path_biased = f"ala2/simulation/o_{100+i}/long_C7ax_{-3.5}/"
        s_C7eq_xv,s_C7eq_fs,s_C7eq_xv_heavy,s_C7eq_fs_heavy = load_data(C7eq_path_biased,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
        s_C7ax_xv,s_C7ax_fs,s_C7ax_xv_heavy,s_C7ax_fs_heavy = load_data(C7ax_path_biased,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
        
        
        data_biased = torch.cat((data_biased,s_C7eq_xv_heavy.to(device),s_C7ax_xv_heavy.to(device)),dim=0)
        data_biased_xv = torch.cat((data_biased_xv,s_C7eq_xv.to(device),s_C7ax_xv.to(device)),dim=0)
    
    data_biased = torch.cat((data_biased,data_biased),dim = 0)
    data_biased_xv = torch.cat((data_biased_xv,data_biased_xv),dim = 0)
    if resample == True:
        vdim_origin = data_biased_xv.shape[1]//2
        #print(vdim_origin)
        data_biased_xv[:,vdim_origin:] = torch.randn(size=(data_biased_xv.shape[0],vdim_origin),
                                                    dtype=torch.float32) * np.sqrt(kbt)/torch.sqrt(mass)
        data_biased[:,xdim:] = data_biased_xv[:,vdim_origin+heavy_dim_indices]


    for i in range(len(gammas)):
        q = qs[i]
        gamma = float(gammas[i])
        data_label = data_labels[i]
        with torch.no_grad():
            q_values = torch.zeros((data_biased.shape[0],1)).to(device)
            NN = 100
            for ii in range(NN):
                vdim_origin = data_biased_xv.shape[1]//2
                #print(vdim_origin)
                data_biased_xv[:,vdim_origin:] = torch.randn(size=(data_biased_xv.shape[0],vdim_origin),
                                                            dtype=torch.float32) * np.sqrt(kbt)/torch.sqrt(mass)
                data_biased[:,xdim:] = data_biased_xv[:,vdim_origin+heavy_dim_indices]
                q_values += q(data_biased)
            q_values /= NN
            iso = (q_values - 0.5)**2<1e-4
            iso = iso.cpu()
            print(q(data_biased[iso.squeeze(),:]))
            iso_xv = data_biased_xv[iso.squeeze(),:]
            dihedrals_phipsi = phipsi(data_biased[iso.squeeze(),:xdim], num_heavy_atoms, phi_group_heavy, psi_group_heavy)
            dihedrals_phitheta = phipsi(data_biased[iso.squeeze(),:xdim], num_heavy_atoms, phi_group_heavy, theta_group_heavy)
            
            mask_phi = torch.abs(dihedrals_phitheta[:,0])<0.5
            mask_theta = torch.abs(dihedrals_phitheta[:,1])<0.5
            mask = mask_phi&mask_theta
            iso_xv = iso_xv[mask,:]
            iso_xv = iso_xv.detach().cpu()
            dihedrals_phipsi = dihedrals_phipsi[mask,:]
            dihedrals_phitheta = dihedrals_phitheta[mask,:]
            dihedrals_phipsi = dihedrals_phipsi.to('cpu').detach().numpy()
            dihedrals_phitheta = dihedrals_phitheta.to('cpu').detach().numpy()
            plt.scatter(dihedrals_phipsi[:,0],dihedrals_phipsi[:,1])
            
            contour_lines = plt.contour(phi_contour_1, psi_contour_1, U_phipsi, levels=10, cmap="viridis")
            plt.clabel(contour_lines, inline=True, fontsize=8)
            plt.xlim([-np.pi, np.pi])
            plt.ylim([-np.pi, np.pi])
            plt.savefig('ala2/fig/temp.png')
            plt.close()
            plt.scatter(dihedrals_phitheta[:,0],dihedrals_phitheta[:,1])
            contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=10, cmap="viridis")
            plt.clabel(contour_lines, inline=True, fontsize=8)
            plt.xlim([-np.pi, np.pi])
            plt.ylim([-np.pi, np.pi])
            plt.savefig('ala2/fig/temp1.png')
            shooting_record = np.zeros((iso_xv.shape[0]))
            print(f"Number of shooting points for gamma={gamma},data_label={data_label}: {iso_xv.shape[0]}")
            start = time.time()
            isopath = f"ala2/simulation/iso/gamma{gammas[i]}_{data_label}/"
            if os.path.exists(isopath)==False:
                os.makedirs(isopath)
            for idx in range(iso_xv.shape[0]):
                if True:
                    isopath_i = isopath + f"point_{idx}/"
                    if os.path.exists(isopath_i)==False:
                        os.makedirs(isopath_i)
                    x = iso_xv[idx,:all_atom_num*3].reshape(-1,3)
                    #print(x)
                    v = iso_xv[idx,all_atom_num*3:].reshape(-1,3)
                    #print(v)
                    write_gro_from_torch(x, isopath_i + f"iso_{idx}.gro",
                                title=parsed["title"],
                                atom_names=parsed["atom_names"],
                                residue_names=parsed["residue_names"],
                                residue_numbers=parsed["residue_numbers"],
                                velocities=v,
                                input_in_angstrom=False,
                                box=parsed["box"] if isinstance(parsed["box"], (list, tuple)) else None)
                '''
                shooting = False
                os.chdir('ala2/simulation')
                gro_file = "iso/iso.gro"
                mdp_file = "iso/nvt_very_short.mdp"
                tpr_file = "iso/nvt_very_short.tpr"
                output_name = "iso/iso"
                plumed_file = "iso/plumed_iso.dat"
                Colvar_name = "iso/COLVAR"
                shooting_file = "iso/shooting_iso.sh"
                r = 10/180*np.pi  
                steps = 10000
                nth_step = -1
                while not shooting:
                    nth_step += 1
                    print(f"  Running short MD step {nth_step+1}")
                    if nth_step > 20:
                        print("  Shooting failed: exceeded maximum number of MD steps")
                        shooting_record[i] = np.nan
                        break
                    write_mdp_file(mdp_file,gamma,nstep=steps,nstxout=1,nstvout=0,nstfout=0,nstenergy=0,nstlog=0)
                    write_plumed_print_file(plumed_file,Colvar_name,stride = 1)
                    write_shooting_file(shooting_file,gro_file, mdp_file,tpr_file,output_name,plumed_file)
                    cmd0 = f"chmod +x {shooting_file}"
                    cmd1 = f"bash {shooting_file}"
                    
                    
                    cmd2 = f"rm -- iso/bck.*"
                    
                    cmd3 = f"rm -- iso/'#'*"
                    cmds = [cmd0, cmd1, cmd2, cmd3]
                    for cmd in cmds:
                        #with open("/dev/null", "wb") as devnull:   
                        #    subprocess.run(cmd, shell=True,stdout=devnull, stderr=devnull)
                        subprocess.run(cmd, shell=True)
                    colvar = read_COLVAR(Colvar_name)
                    phipsi = colvar[:,1:3]

                    dC7eq = phipsi - c_C7eq
                    dC7ax = phipsi - c_C7ax
                    mask_eq = (dC7eq[:,0]**2 + dC7eq[:,1]**2) < r**2
                    mask_ax = (dC7ax[:,0]**2 + dC7ax[:,1]**2) < r**2
                    idxs_eq = np.nonzero(mask_eq)[0]
                    
                    if idxs_eq.shape[0] > 0:
                        idx_eq = idxs_eq[0]
                        shooting = True
                    else:
                        idx_eq = 2* steps
                    idxs_ax = np.nonzero(mask_ax)[0]
                    if idxs_ax.shape[0] > 0:
                        idx_ax = idxs_ax[0]
                        shooting = True
                    else:
                        idx_ax = 2*steps

                    if shooting:
                        if idx_eq < idx_ax:
                            shooting_record[i] = 0
                        else:
                            shooting_record[i] = 1
                            
                    



                os.chdir('../..')
            end = time.time()
            print(f"Time taken for shooting simulations: {(end - start)/iso_xv.shape[0]} seconds/points")
            '''
            