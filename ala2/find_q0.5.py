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

epsilon = 1e-10

gamma_o = 10040
kbt = 300 * 0.0083144621  # kBT in kcal/mol 
num_heavy_atoms = heavy_atom_indices.shape[0]
xdim = heavy_atom_indices.shape[0] * 3
vdim = heavy_atom_indices.shape[0] * 3
xdim_reduce = 4
vdim_reduce = 4


gamma_data_label_file = "ala2/distilling_gamma_data_label.txt"
gammas, data_labels = get_gamma_data_label(gamma_data_label_file)
print(gammas,data_labels)
#data_label = 'long'
#data_label = 'long_1'
#data_label = 'long_2'
#data_label = 'long_3'
#data_label = 'biased'
#data_label = 'all'

#xdim_reduce = 45
#vdim_reduce = 45  
layers = [xdim_reduce+vdim_reduce,8,64,64,64,64,8,1]
layers_0 = [xdim_reduce, 64, 64, 1]
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

highT_path = "ala2/simulation/1500K/"  # Working directory for intermediate files
C7eq_path = "ala2/simulation/long_C7eq/"
C7ax_path = "ala2/simulation/long_C7ax/"

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


part = 0.1
C7eq_xv_heavy = C7eq_xv_heavy[:int(C7eq_xv_heavy.shape[0]*part)]
C7ax_xv_heavy = C7ax_xv_heavy[:int(C7ax_xv_heavy.shape[0]*part)]
label_a = torch.zeros(C7eq_xv_heavy.shape[0], dtype=torch.float32)
label_b = torch.ones(C7ax_xv_heavy.shape[0], dtype=torch.float32)
data_boundary = torch.cat((C7eq_xv_heavy, C7ax_xv_heavy), dim=0)
label_boundary = torch.cat((label_a, label_b), dim=0).unsqueeze(1)

data_T = highT_xv_heavy
dU_T = -highT_fs_heavy 





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


heavy_atom_mass = heavy_atom_mass.to(device)




# q = NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=psi_group_heavy, activation='sigmoid')
#q = NNd2_45(layer_sizes=layers,n_atoms=num_heavy_atoms, activation='sigmoid')
#q = NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy, activation='sigmoid')
#q.load_state_dict(torch.load(f"ala2/model/gamma{gamma}_kbt{kbt}_{data_label}.pth"))
model_path_o = f"ala2/model/distilling_gamma0.20005/gamma0.20005_highT_subtrain_2.pth"
config_path_o = f"ala2/config/distilling_gamma0.20005/gamma0.20005_highT_subtrain_2.txt"
q0 = load_model_phipsi(model_path_o,config_path_o)



figname_q0 = f'ala2/fig/q0_1.png'
filename_dq0 = f'ala2/fig/dq0_1.png'

#draw_q0_dq0(q0,360,data_biased,figname_q0,filename_dq0,device=device)

def shooting_simulation(workdir,num_simulations,
                        gro_file,
                        mdp_file,
                        tpr_file,
                        output_name,
                        plumed_file,
                        Colvar_name,
                        shooting_file,
                        c_C7eq,
                        c_C7ax,
                        gamma,
                        nsteps,
                        max_steps = 100):
    origin_directory = os.getcwd()
    os.chdir(workdir)
    shooting_record = 0
    N_simulation = 0
    r = 10/180*np.pi  
    
    for i in range(num_simulations):
        shooting = False
        nth_step = -1
        while not shooting:
            nth_step += 1
            print(f"  Running short MD step {nth_step+1}")
            if nth_step > max_steps:
                break
            write_mdp_file(mdp_file,gamma,nstep=nsteps,nstxout=1,nstvout=0,nstfout=0,nstenergy=0,nstlog=0)
            write_plumed_print_file(plumed_file,Colvar_name,stride = 1)
            write_shooting_file(shooting_file,gro_file, mdp_file,tpr_file,output_name,plumed_file)
            cmd0 = f"chmod +x {shooting_file}"
            cmd1 = f"bash {shooting_file}"
            
            
            cmd2 = f"rm -- iso/bck.*"
            
            cmd3 = f"rm -- iso/'#'*"
            cmds = [cmd0, cmd1, cmd2, cmd3]
            for cmd in cmds:
                with open("/dev/null", "wb") as devnull:   
                    subprocess.run(cmd, shell=True,stdout=devnull, stderr=devnull)
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
                N_simulation += 1
                if idx_eq < idx_ax:
                    shooting_record += 0
                else:
                    shooting_record += 1
                
        



    os.chdir(origin_directory)
    return shooting_record,N_simulation





filename_vs = f"ala2/model/vs_{kbt}.txt"
vs = np.loadtxt(filename_vs)
vs = torch.from_numpy(vs.astype(np.float32)).to(device)

if True:
    qs = []
    figname_mqs = []
    figname_q_datas = []
    figname_mq_datas = []
    figname_data_logps = []
    figname_q_vslices = []
    for gamma, data_label in zip(gammas, data_labels):
        qs.append(NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy, activation='sigmoid'))
        qs[-1].load_state_dict(torch.load(f"ala2/model/gamma{gamma}_kbt{kbt}_{data_label}.pth",map_location=torch.device('cpu')))
        qs[-1].to(device)
        if not os.path.exists(f"ala2/fig/gamma{gamma}"):
            os.makedirs(f"ala2/fig/gamma{gamma}")

        figname_mqs.append(f'ala2/fig/gamma{gamma}/mean_q_gamma{gamma}_{data_label}.png')
        figname_q_datas.append(f'ala2/fig/gamma{gamma}/q_data_gamma{gamma}_{data_label}.png')
        figname_mq_datas.append(f'ala2/fig/gamma{gamma}/mq_data_gamma{gamma}_{data_label}.png')
        figname_data_logps.append(f'ala2/fig/gamma{gamma}/data_logp_gamma{gamma}_{data_label}.png')
        figname_q_vslices.append(f'ala2/fig/gamma{gamma}/vslice_committor_gamma{gamma}_{data_label}.png')
    data_T = data_T.to(device)
    for i in range(len(gammas)):
        q = qs[i]
        gamma = float(gammas[i])
        data_label = data_labels[i]
        with torch.no_grad():
            q_values = q(data_T)
            iso = (q_values - 0.5)**2<25 * 10e-6
            iso = iso.cpu()
            iso_xv = highT_xv[iso.squeeze(),:].detach().cpu()
            shooting_record = np.zeros((iso_xv.shape[0]))
            print(f"Number of shooting points for gamma={gamma},data_label={data_label}: {iso_xv.shape[0]}")
            start = time.time()
            isopath = f"ala2/simulation/iso/gamma{gamma}_{data_label}/"
            if os.path.exists(isopath)==False:
                os.makedirs(isopath)
            for i in range(iso_xv.shape[0]):
                isopath_i = isopath + f"point_{i}/"
                if os.path.exists(isopath_i)==False:
                    os.makedirs(isopath_i)
                x = iso_xv[i,:all_atom_num*3].reshape(-1,3)
                #print(x)
                v = iso_xv[i,all_atom_num*3:].reshape(-1,3)
                #print(v)
                write_gro_from_torch(x, isopath_i + f"iso_{i}.gro",
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
            