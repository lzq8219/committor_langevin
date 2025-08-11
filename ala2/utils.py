import numpy as np
import os
import sys
import subprocess
import matplotlib.pyplot as plt

from multiprocessing import Pool


def restrained_simulation(grid_point):
    phi_deg, psi_deg, phi, psi = grid_point
    dir_name = f"phi_{phi_deg}_psi_{psi_deg}"
    os.makedirs(dir_name, exist_ok=True)

    # Generate plumed.dat
    plumed_content_initial = f"""
MOLINFO STRUCTURE=./ala2.pdb

phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2
restraint: RESTRAINT ARG=phi,psi AT={phi},{psi} KAPPA=100.0,100.0

PRINT ARG=phi,psi,restraint.bias FILE={dir_name}/COLVAR_initial STRIDE=100
"""

    plumed_content = f"""
MOLINFO STRUCTURE=./ala2.pdb

phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2
restraint: RESTRAINT ARG=phi,psi AT={phi},{psi} KAPPA=1000.0,1000.0

PRINT ARG=phi,psi,restraint.bias FILE={dir_name}/COLVAR_restrained STRIDE=100
"""
    with open(f"{dir_name}/plumed.dat", "w") as f:
        f.write(plumed_content)

    with open(f"{dir_name}/plumed_initial.dat", "w") as f:
        f.write(plumed_content_initial)

    # Copy input files

    # Run simulation
    subprocess.run(
        f"gmx_mpi mdrun -s nvt.tpr -plumed {dir_name}/plumed_initial.dat -deffnm {dir_name}/initial -v",
        shell=True,
    )
    subprocess.run(
        f"gmx_mpi grompp -f nvt.mdp -c {dir_name}/initial.gro -r {dir_name}/initial.gro -p topol.top -o {dir_name}/restrained.tpr",
        shell=True,
    )
    subprocess.run(
        f"gmx_mpi mdrun -s {dir_name}/restrained.tpr -plumed {dir_name}/plumed.dat -deffnm {dir_name}/restrained -v",
        shell=True,
    )

# Create grid


def plot_colvar(filename, figname):
    colvar = np.loadtxt(filename, comments="#")
    deg_to_rad = np.pi / 180.0
    plt.figure(figsize=(10, 6))
    plt.scatter(colvar[:, 1]/deg_to_rad, colvar[:, 2]/deg_to_rad, c=colvar[:, 3], alpha=0.5)
    plt.colorbar(label='Restraint Bias')
    plt.xlabel('phi (rad)')
    plt.ylabel('psi (rad)')
    plt.title('COLVAR Analysis')
    plt.savefig(figname)
    plt.close()

def FES_gradient(grid_point):
    phi_deg, psi_deg, phi, psi = grid_point
    dir_name = f"phi_{phi_deg}_psi_{psi_deg}"
    colvar = np.loadtxt(f"{dir_name}/COLVAR_restrained", comments="#")
    phi_restrained = colvar[:, 1]
    psi_restrained = colvar[:, 2]
    bias = colvar[:, 3]
    dphi = phi_restrained - phi
    dpsi = psi_restrained - psi

    while np.any(dphi >= np.pi):
        dphi[dphi >= np.pi] -= 2 * np.pi
    while np.any(dphi <= -np.pi):
        dphi[dphi <= -np.pi] += 2 * np.pi
    while np.any(dpsi >= np.pi):
        dpsi[dpsi >= np.pi] -= 2 * np.pi
    while np.any(dpsi <= -np.pi):
        dpsi[dpsi <= -np.pi] += 2 * np.pi

    gradient_phi = -1000.0 * np.mean(dphi)
    gradient_psi = -1000.0 * np.mean(dpsi)

    return gradient_phi, gradient_psi



if __name__ == "__main__":
    step = 10
    deg_to_rad = np.pi / 180.0
    phi_angles = np.arange(-180, 181, step)
    psi_angles = np.arange(-180, 181, step)
    grid = [(phi_deg, psi_deg, phi_deg * deg_to_rad, psi_deg * deg_to_rad)
            for phi_deg in phi_angles for psi_deg in psi_angles]

    # Run in parallel
    '''
    for g in grid:
        restrained_simulation(g)
    '''

    '''
    for g in grid:
        phi_deg, psi_deg, _, _ = g
        dir_name = f"phi_{phi_deg}_psi_{psi_deg}"
        plot_colvar(
            f"{dir_name}/COLVAR_restrained",
            f"{dir_name}/COLVAR_restrained.png")
    '''

    
    output_file = "fes_gradients.csv"
    with open(output_file, "w") as file:
    # Write a header (optional)
        file.write("phi_deg, psi_deg, phi, psi, grad_phi, grad_psi\n")
    
    # Loop through the grid and calculate gradients
        for g in grid:
            phi_deg, psi_deg, phi, psi = g
            dir_name = f"phi_{phi_deg}_psi_{psi_deg}"
            grad_phi, grad_psi = FES_gradient(g)
            grad_phi, grad_psi = FES_gradient(g)
            
            # Write the data line by line
            file.write(f"{phi_deg}, {psi_deg}, {phi}, {psi}, {grad_phi}, {grad_psi}\n")
    

    '''
    data = np.loadtxt("fes_gradients.csv", delimiter=",", skiprows=1, usecols=(0, 1, 4, 5), unpack=True)
    phi_deg, psi_deg, grad_phi, grad_psi = data
    X, Y = np.meshgrid(phi_deg, psi_deg)
    DX,DY = np.meshgrid(grad_phi, grad_psi)
    plt.figure(figsize=(10, 6))
    plt.quiver(X, Y, DX, DY,np.sqrt(DX**2 + DY**2), cmap='viridis',scale = 1)
    plt.xlabel('phi (degrees)')
    plt.ylabel('psi (degrees)')
    plt.title('FES Gradient Field')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.grid()
    plt.savefig("fes_gradient_field.png")
    plt.close()
    '''