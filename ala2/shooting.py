import argparse
import sys
import os
import shutil
import torch
import numpy as np

import subprocess

from gro import parsed,write_gro_from_torch
import time

def write_mdp_file(filename, 
                   gamma ,
                   nstep,
                   nstxout = 200,
                   nstvout = 200, 
                   nstenergy = 200, 
                   nstlog = 200, 
                   nstfout = 200):
    if nstxout == 0:
        nstxout = nstep
    if nstvout == 0:
        nstvout = nstep 
    if nstenergy == 0:
        nstenergy = nstep
    if nstlog == 0:
        nstlog = nstep
    if nstfout == 0:
        nstfout = nstep
    file = f'''
; Run control parameters
integrator               = sd                ; Use molecular dynamics integrator
nsteps                   = {nstep}         ; 20*500 ps (10,000 steps with a 2 fs timestep)
dt                       = 0.002             ; Time step of 2 fs

; Output control
nstxout                 = {nstxout}                ; Save coordinates every 1 ps
nstvout                 = {nstvout}                ; Save velocities every 1 ps
nstenergy               = {nstenergy}                ; Save energies every 1 ps
nstlog                  = {nstlog}                ; Save log data every 1 ps
nstfout                 = {nstfout}  

; Neighbor searching
cutoff-scheme           = Verlet             ; Verlet cutoff scheme
nstlist                 = 10                 ; Neighbor list updated every 10 steps
rlist                   = 0.9                ; Short-range neighbor list cutoff (nm)

; Electrostatics and VDW
coulombtype             = cutoff                ; Particle Mesh Ewald for Coulomb interactions
rcoulomb                = 0.9                ; Real-space cutoff for Coulomb interactions (nm)
vdwtype                 = cutoff             ; Cutoff for van der Waals interactions
rvdw                    = 0.9                ; Real-space cutoff for van der Waals interactions (nm)
fourierspacing          = 0.1                ; PME grid spacing (nm)

; Temperature coupling
tc-grps                 = System             ; Single coupling group for the whole system
tau_t                   = {1/gamma:.6f}                ; Temperature relaxation time constant (ps)
ref_t                   = 300               ; Target temperature (K)

; Pressure coupling
pcoupl                  = no                 ; Disable pressure coupling for NVT ensemble

; Periodic boundary conditions
pbc                     = xyz                ; Fully periodic boundary conditions

; Miscellaneous
gen_vel                 = no                ; Generate velocities at the beginning
'''
    with open(filename, 'w') as f:
        f.write(file)


def write_shooting_file(filename,gro_file, mdp_file,tpr_file,output_name,plumed_file):
    file = f'''
# Run the first command
gmx_mpi grompp -f {mdp_file} -c {gro_file} -p topol.top -o {tpr_file}
OMP_NUM_THREADS=1 gmx_mpi mdrun -s {tpr_file} -deffnm {output_name} 
plumed driver --mf_trr {output_name}.trr --plumed {plumed_file}
'''
    with open(filename, 'w') as f:
        f.write(file)

def write_plumed_print_file(filename,Colvar_name,stride = 100):
    file = f'''
    MOLINFO STRUCTURE=./ala2.pdb

phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2
theta: TORSION ATOMS=6,5,7,9

#C7eq phi=-1.39626,psi=1.30899
#C7ax phi=1.30899,psi=-1.0471 

PRINT ARG=phi,psi,theta FILE={Colvar_name} STRIDE={stride}
'''
    with open(filename, 'w') as f:
        f.write(file)

def read_COLVAR(filename):
    file_path = filename  # Replace with your actual file name

    # Read the file and filter out lines starting with '#!'
    with open(file_path, "r") as file:
        data_lines = [line for line in file if not line.startswith("#")]

    # Convert the remaining lines into a NumPy array
    data = np.loadtxt(data_lines)

    return data


def shooting_simulation(workdir,num_simulations,
                        file_dir,
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
    shooting_record = 0
    N_simulation = 0
    r = 10/180*np.pi  
    
    for i in range(num_simulations):
        shooting = False
        nth_step = -1
        shutil.copy(gro_file, output_name + ".gro")
        write_mdp_file(mdp_file,gamma,nstep=nsteps,nstxout=1,nstvout=0,nstfout=0,nstenergy=0,nstlog=0)
        write_plumed_print_file(plumed_file,Colvar_name,stride = 1)
        write_shooting_file(shooting_file,output_name+".gro", mdp_file,tpr_file,output_name,plumed_file)
        while not shooting:
            nth_step += 1
            print(f"  Running short MD step {nth_step+1}")
            if nth_step > max_steps:
                break
            
            cmd0 = f"chmod +x {shooting_file}"
            cmd1 = f"bash {shooting_file}"
            
            
            cmd2 = f"rm -- {file_dir}bck.*"
            
            cmd3 = f"rm -- {file_dir}'#'*"
            cmds = [cmd0, cmd1, cmd2, cmd3]
            for cmd in cmds:
                # print(f"Executing: {cmd}")
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
                idx_eq = 2* nsteps
            idxs_ax = np.nonzero(mask_ax)[0]
            if idxs_ax.shape[0] > 0:
                idx_ax = idxs_ax[0]
                shooting = True
            else:
                idx_ax = 2*nsteps

            if shooting:
                N_simulation += 1
                if idx_eq < idx_ax:
                    shooting_record += 0
                else:
                    shooting_record += 1
                
        

    with open(file_dir+'result', 'w') as f:
        f.write(f'Shooting record: {shooting_record}\n')
        f.write(f'Number of simulations: {N_simulation}\n')


    return shooting_record,N_simulation

def parse_args():
    p = argparse.ArgumentParser(description="Shooting analysis")
    p.add_argument("--idx", "-i", type=int, required=True)
    p.add_argument("--dir","-d", type=str, default=".", help="Working directory where commands are executed")
    p.add_argument("--gamma", "-g", type=float, required=True, help="Working directory where commands are executed")
    p.add_argument("--num_simulations", "-n", type=int, required=True, help="data label")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    idx = args.idx
    dir = args.dir  +f"point_{idx}/"
    workdir = "."

    gro_file = dir + f"iso_{idx}.gro"
    mdp_file = dir +"iso.mdp"
    tpr_file = dir +"iso.tpr"
    output_name = dir +"iso"
    plumed_file = dir +"plumed.dat"
    Colvar_name = dir +"COLVAR"
    shooting_file = dir +"shooting.sh"
    c_C7eq = np.array([-1.46, 1.3305264])
    c_C7ax = np.array([1.01, -0.71])
    gamma = args.gamma
    nsteps = 100000
    num_simulations = args.num_simulations

    shooting_record,N_simulation = shooting_simulation(workdir,num_simulations,
                        dir,       
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
                        nsteps)
    