

from multiprocessing import Pool
from nn import FunctionModelWithDescriptor
import numpy as np
import os
import sys
import subprocess
import matplotlib.pyplot as plt
import torch
import itertools

origin_directory = os.getcwd()
# print(f"Origin directory: {origin_directory}")
model_directory = os.path.join(origin_directory, 'ala2')
src_directory = os.path.join(origin_directory, 'src')
sys.path.append(src_directory)
sys.path.append(model_directory)


phi_group = torch.tensor([5, 7, 9, 15], dtype=torch.int32) - 1
psi_group = torch.tensor([7, 9, 15, 17], dtype=torch.int32) - 1
theta_group = torch.tensor([6, 5, 7, 9], dtype=torch.int32) - 1
heavy_atom_indices = torch.tensor(
    [2, 5, 6, 7, 9, 11, 15, 16, 17, 19], dtype=torch.int32) - 1  # Exclude hydrogens
heavy_dim_indices = np.stack([(3 * x, 3 * x + 1, 3 * x + 2)
                             for x in heavy_atom_indices], axis=0).reshape(-1)
phi_group_heavy = torch.nonzero(heavy_atom_indices.unsqueeze(
    0) == phi_group.unsqueeze(1), as_tuple=True)[1]
psi_group_heavy = torch.nonzero(heavy_atom_indices.unsqueeze(
    0) == psi_group.unsqueeze(1), as_tuple=True)[1]
theta_group_heavy = torch.nonzero(heavy_atom_indices.unsqueeze(
    0) == theta_group.unsqueeze(1), as_tuple=True)[1]
all_atom_num = 22

# C7eq phi=-1.39626,psi=1.30899
# C7ax phi=1.30899,psi=-1.0471
c_C7eq = np.array([[-1.39626, 1.30899]])
c_C7ax = np.array([[1.30899, -1.0471]])


def rate(model, data, weight, args, device, xdim, vdim):
    # data and weight should be sampled from the equilibrium distribution
    data = data.to(device)
    weight = weight.to(device)
    data.requires_grad_(True)
    kbt = args['kbt']
    gamma = args['gamma']
    xdim = args['xdim']

    qqq = model(data)
    with torch.no_grad():
        gradients = torch.autograd.grad(outputs=qqq, inputs=data,
                                        grad_outputs=torch.ones_like(qqq),
                                        create_graph=False, retain_graph=False)[0]
    if weight.shape is not (data.shape[0], 1):
        weight = weight.unsqueeze(dim=1)
    grad_v = gradients[:, xdim:]

    return gamma * kbt * torch.sum(weight * (grad_v**2))


def plot_loss(loss_list, b_loss_list, tot_loss_list, pinn_loss_list, fig_file):
    t = np.arange(len(loss_list))  # Time values


# Create a figure with 3 subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

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

    axs[3].plot(t, pinn_loss_list, label='Pinn Loss', color='red')
    axs[3].set_title('Pinn Loss vs Time')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Loss')
    axs[3].legend()
    axs[3].grid()

    plt.savefig(fig_file, dpi=300, bbox_inches='tight')


def restrained_simulation(grid_point):
    phi_deg, psi_deg, phi, psi = grid_point
    os.chdir('./ala2/simulation/')
    dir_name = f"constrained/phi_{phi_deg}_psi_{psi_deg}"
    os.makedirs(dir_name, exist_ok=True)

    # Generate plumed.dat
    plumed_content_initial = f"""
MOLINFO STRUCTURE=./ala2.pdb

phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2
theta: TORSION ATOMS=6,5,7,9
restraint: RESTRAINT ARG=phi,theta AT={phi},{psi} KAPPA=100.0,100.0

PRINT ARG=phi,theta,restraint.bias FILE={dir_name}/COLVAR_initial STRIDE=100
"""

    plumed_content = f"""
MOLINFO STRUCTURE=./ala2.pdb

phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2
theta: TORSION ATOMS=6,5,7,9
restraint: RESTRAINT ARG=phi,theta AT={phi},{psi} KAPPA=1000.0,1000.0

PRINT ARG=phi,theta,restraint.bias FILE={dir_name}/COLVAR_restrained STRIDE=100
"""
    with open(f"{dir_name}/plumed.dat", "w") as f:
        f.write(plumed_content)

    with open(f"{dir_name}/plumed_initial.dat", "w") as f:
        f.write(plumed_content_initial)

    # Copy input files

    # Run simulation
    subprocess.run(
        f"gmx_mpi mdrun -s nvt_short.tpr -plumed {dir_name}/plumed_initial.dat -deffnm {dir_name}/initial -v",
        shell=True,
    )
    subprocess.run(
        f"gmx_mpi grompp -f nvt_constrained.mdp -c {dir_name}/initial.gro -r {dir_name}/initial.gro -p topol.top -o {dir_name}/restrained.tpr",
        shell=True,
    )
    subprocess.run(
        f"gmx_mpi mdrun -s {dir_name}/restrained.tpr -plumed {dir_name}/plumed.dat -deffnm {dir_name}/restrained -v",
        shell=True,
    )
    os.chdir(origin_directory)

# Create grid


def plot_colvar(filename, figname):
    colvar = np.loadtxt(filename, comments="#")
    deg_to_rad = np.pi / 180.0
    plt.figure(figsize=(10, 6))
    plt.scatter(colvar[:, 1] / deg_to_rad, colvar[:, 2] /
                deg_to_rad, c=colvar[:, 3], alpha=0.5)
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


def extract_trr_data(trr_file, tpr_file, working_path, xfile, vfile, ffile):
    """
    Extract positions (x), velocities (v), and forces (f) from a .trr file
    using GROMACS tools and write the data to a .txt file.

    Args:
        trr_file (str): Path to the .trr trajectory file.
        tpr_file (str): Path to the .tpr topology file.
        output_file (str): Path to the output .txt file.
    """
    # Output files for intermediate data
    positions_file = working_path + xfile
    velocities_file = working_path + vfile
    forces_file = working_path + ffile

    try:
        # Extract positions (x) using GROMACS
        subprocess.run(
            ["echo 0 |", "gmx_mpi", "traj", "-f", trr_file,
                "-s", tpr_file, "-ox", positions_file],
            check=True
        )
        print(f"Positions extracted to {positions_file}")

        # Extract velocities (v) using GROMACS
        subprocess.run(
            ["echo 0 |", "gmx_mpi", "traj", "-f", trr_file,
                "-s", tpr_file, "-ov", velocities_file],
            check=True
        )
        print(f"Velocities extracted to {velocities_file}")

        # Extract forces (f) using GROMACS
        subprocess.run(
            ["echo 0 |", "gmx_mpi", "traj", "-f", trr_file,
                "-s", tpr_file, "-of", forces_file],
            check=True
        )
        print(f"Forces extracted to {forces_file}")

        # Write the extracted data to a single .txt file

    except subprocess.CalledProcessError as e:
        print(f"Error while running GROMACS command: {e}")
    except FileNotFoundError:
        print("GROMACS tools are not installed or not in PATH. Please install GROMACS and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_xvg(file_path):
    """
    Reads a GROMACS .xvg file and loads the numerical data into a NumPy array.

    Args:
        file_path (str): Path to the .xvg file.

    Returns:
        np.ndarray: A NumPy array containing the numerical data.
    """
    try:
        # Use numpy.loadtxt to load the file, skipping lines that start with #
        # or @
        data = np.loadtxt(file_path, comments=("#", "@"))
        return data[:, 1:].copy()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def compute_dihedral_cossin(coords):
    """
    Computes dihedral angles for an n*4*3 tensor of coordinates.

    Args:
        coords (torch.Tensor): Tensor of shape (n, 4, 3), where each row contains
                               the coordinates of 4 atoms.
    Returns:
        torch.Tensor: Tensor of shape (n,), containing dihedral angles in radians.
    """
    # Vectors between adjacent atoms
    b1 = coords[:, 1] - coords[:, 0]  # Vector between point 1 and 2
    b2 = coords[:, 2] - coords[:, 1]  # Vector between point 2 and 3
    b3 = coords[:, 3] - coords[:, 2]  # Vector between point 3 and 4

    # Normalize b2 (central bond)
    b2_norm1 = b2 / b2.norm(dim=1, keepdim=True)

    # Calculate perpendicular vectors to the planes
    n1 = torch.cross(b1, b2, dim=1)  # Plane 1: Points 1, 2, 3
    n2 = torch.cross(b2, b3, dim=1)             # Plane 2: Points 2, 3, 4

    # Normalize n1 and n2
    n1_norm1 = n1 / n1.norm(dim=1, keepdim=True)

    n2_norm1 = n2 / n2.norm(dim=1, keepdim=True)

    # Calculate the angle between v1 and v2
    cos_angle = torch.sum(n1_norm1 * n2_norm1, dim=1)

    # Calculate the sign of the angle using the direction of b2

    # Dihedral angle (radians)
    sin_angle = torch.sum(
        b2_norm1 *
        torch.cross(
            n1_norm1,
            n2_norm1,
            dim=1),
        dim=1)
    dihedral = torch.atan2(sin_angle,cos_angle)

    return cos_angle, sin_angle, dihedral


def compute_dihedral_cossin_1(coords):
    """
    Computes dihedral angles for an n*4*3 tensor of coordinates.

    Args:
        coords (torch.Tensor): Tensor of shape (n, 4, 3), where each row contains
                               the coordinates of 4 atoms.
    Returns:
        torch.Tensor: Tensor of shape (n,), containing dihedral angles in radians.
    """
    # Vectors between adjacent atoms
    b1 = coords[:, 1] - coords[:, 0]  # Vector between point 1 and 2
    b2 = coords[:, 2] - coords[:, 1]  # Vector between point 2 and 3
    b3 = coords[:, 3] - coords[:, 2]  # Vector between point 3 and 4

    # Normalize b2 (central bond)
    b2_norm1 = b2 / b2.norm(dim=1, keepdim=True)

    # Calculate perpendicular vectors to the planes
    n1 = torch.cross(b1, b2, dim=1)  # Plane 1: Points 1, 2, 3
    n2 = torch.cross(b2, b3, dim=1)             # Plane 2: Points 2, 3, 4

    # Normalize n1 and n2
    n1_norm1 = n1 / n1.norm(dim=1, keepdim=True)

    n2_norm1 = n2 / n2.norm(dim=1, keepdim=True)

    # Calculate the angle between v1 and v2
    cos_angle = torch.sum(n1_norm1 * n2_norm1, dim=1)

    # Calculate the sign of the angle using the direction of b2

    # Dihedral angle (radians)
    sin_angle = torch.sum(
        b2_norm1 *
        torch.cross(
            n1_norm1,
            n2_norm1,
            dim=1),
        dim=1)

    return cos_angle, sin_angle


def compute_dihedral_dcossindt(coords, v_coords):
    """
    Computes dihedral angles for an n*4*3 tensor of coordinates.

    Args:
        coords (torch.Tensor): Tensor of shape (n, 4, 3), where each row contains
                               the coordinates of 4 atoms.
    Returns:
        torch.Tensor: Tensor of shape (n,), containing dihedral angles in radians.
    """
    # Vectors between adjacent atoms
    b1 = coords[:, 1] - coords[:, 0]  # Vector between point 1 and 2
    b2 = coords[:, 2] - coords[:, 1]  # Vector between point 2 and 3
    b3 = coords[:, 3] - coords[:, 2]  # Vector between point 3 and 4
    v_b1 = v_coords[:, 1] - v_coords[:, 0]  # Vector between point 1 and 2
    v_b2 = v_coords[:, 2] - v_coords[:, 1]  # Vector between point 2 and 3
    v_b3 = v_coords[:, 3] - v_coords[:, 2]  # Vector between point 3 and 4

    # Normalize b2 (central bond)
    b2_norm1 = b2 / b2.norm(dim=1, keepdim=True)
    v_b2_norm1 = (v_b2 - b2_norm1 * torch.sum(v_b2 * b2_norm1,
                  dim=1, keepdim=True)) / b2.norm(dim=1, keepdim=True)

    # Calculate perpendicular vectors to the planes
    n1 = torch.cross(b1, b2, dim=1)  # Plane 1: Points 1, 2, 3
    n2 = torch.cross(b2, b3, dim=1)             # Plane 2: Points 2, 3, 4
    v_n1 = torch.cross(v_b1, b2, dim=1) + torch.cross(b1, v_b2, dim=1)
    v_n2 = torch.cross(v_b2, b3, dim=1) + torch.cross(b2, v_b3, dim=1)

    # Normalize n1 and n2
    n1_norm1 = n1 / n1.norm(dim=1, keepdim=True)
    v_n1_norm1 = (v_n1 - n1_norm1 * torch.sum(v_n1 * n1_norm1,
                  dim=1, keepdim=True)) / n1.norm(dim=1, keepdim=True)

    n2_norm1 = n2 / n2.norm(dim=1, keepdim=True)
    v_n2_norm1 = (v_n2 - n2_norm1 * torch.sum(v_n2 * n2_norm1,
                  dim=1, keepdim=True)) / n2.norm(dim=1, keepdim=True)

    # Calculate the angle between v1 and v2
    cos_angle = torch.sum(n1_norm1 * n2_norm1, dim=1)
    dcos_angle_dt = torch.sum(
        v_n1_norm1 *
        n2_norm1 +
        n1_norm1 *
        v_n2_norm1,
        dim=1)

    # Calculate the sign of the angle using the direction of b2

    # Dihedral angle (radians)
    sin_angle = torch.sum(
        b2_norm1 *
        torch.cross(
            n1_norm1,
            n2_norm1,
            dim=1),
        dim=1)
    dsin_angle_dt = torch.sum(v_b2_norm1 * torch.cross(n1_norm1, n2_norm1, dim=1) +
                              b2_norm1 * torch.cross(v_n1_norm1, n2_norm1, dim=1) +
                              b2_norm1 * torch.cross(n1_norm1, v_n2_norm1, dim=1), dim=1)

    return cos_angle, sin_angle, dcos_angle_dt, dsin_angle_dt


def phipsi(x_and_v: torch.float32, n_atoms, phi_group, psi_group):
    x = torch.reshape(x_and_v, (x_and_v.shape[0], n_atoms, 3))
    x_phi_group = x[:, phi_group, :]
    x_psi_group = x[:, psi_group, :]


    sinphi, cosphi, phi = compute_dihedral_cossin(x_phi_group)
    sinpsi, cospsi, psi = compute_dihedral_cossin(x_psi_group)

    # print(sinphi.shape,cosphi.shape,sinpsi.shape,cospsi.shape,dsinphidt.shape,dcosphidt.shape,dsinpsidt.shape,dcospsidt.shape)
    descriptors = torch.stack((phi, psi), dim=1)
    # print(descriptors)
    return descriptors


class NNphipsi(FunctionModelWithDescriptor):
    def __init__(self, layer_sizes, n_atoms, phi_group,
                 psi_group, activation='sigmoid'):
        super(
            NNphipsi,
            self).__init__(
            layer_sizes,
            activation=activation,
            using_descriptor=True)
        self.phi_group = phi_group
        self.psi_group = psi_group
        self.n_atoms = n_atoms

    def descriptor(self, x_and_v: torch.float32):
        x_and_v_n_d_3 = torch.reshape(
            x_and_v, (x_and_v.shape[0], self.n_atoms * 2, 3))
        x, v = x_and_v_n_d_3[:, :self.n_atoms,
                             :], x_and_v_n_d_3[:, self.n_atoms:, :]
        x_phi_group = x[:, self.phi_group, :]
        x_psi_group = x[:, self.psi_group, :]
        v_phi_group = v[:, self.phi_group, :]
        v_psi_group = v[:, self.psi_group, :]

        cosphi, sinphi, dcosphidt, dsinphidt = compute_dihedral_dcossindt(
            x_phi_group, v_phi_group)
        cospsi, sinpsi, dcospsidt, dsinpsidt = compute_dihedral_dcossindt(
            x_psi_group, v_psi_group)

        # print(sinphi.shape,cosphi.shape,sinpsi.shape,cospsi.shape,dsinphidt.shape,dcosphidt.shape,dsinpsidt.shape,dcospsidt.shape)
        descriptors = torch.stack(
            (sinphi,
             cosphi,
             sinpsi,
             cospsi,
             dsinphidt,
             dcosphidt,
             dsinpsidt,
             dcospsidt),
            dim=1)
        # print(descriptors)
        return descriptors


class NNphipsi_overdamped(FunctionModelWithDescriptor):
    def __init__(self, layer_sizes, n_atoms, phi_group,
                 psi_group, activation='sigmoid',output_scale=1):
        super(
            NNphipsi_overdamped,
            self).__init__(
            layer_sizes,
            activation=activation,
            using_descriptor=True,
            output_scale=output_scale)
        self.phi_group = phi_group
        self.psi_group = psi_group
        self.n_atoms = n_atoms


    def descriptor(self, x_and_v: torch.float32):
        x_and_v_n_d_3 = torch.reshape(
            x_and_v, (x_and_v.shape[0], self.n_atoms, 3))
        x = x_and_v_n_d_3
        x_phi_group = x[:, self.phi_group, :]
        x_psi_group = x[:, self.psi_group, :]

        cosphi, sinphi = compute_dihedral_cossin_1(x_phi_group)
        cospsi, sinpsi = compute_dihedral_cossin_1(x_psi_group)

        # print(sinphi.shape,cosphi.shape,sinpsi.shape,cospsi.shape,dsinphidt.shape,dcosphidt.shape,dsinpsidt.shape,dcospsidt.shape)
        descriptors = torch.stack((sinphi, cosphi, sinpsi, cospsi), dim=1)
        # print(descriptors)
        return descriptors

    def d_forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x*self.output_scale

    def save(self, examples, config_filename,
             model_filename, is_description=False):
        model1 = NNphipsi_overdamped_1(
            self.layer_sizes,
            self.n_atoms,
            self.phi_group,
            self.psi_group,
            activation=self.activation).to(
            torch.device('cpu'))
        model1.load_state_dict(self.state_dict())
        if not is_description:
            examples = self.descriptor(examples).to(
                torch.device('cpu')).detach()
        else:
            examples = examples.to(torch.device('cpu')).detach()
        traced_script_module = torch.jit.trace(
            model1.forward, example_inputs=examples)
        traced_script_module.save(model_filename)
        print(f"Model saved to {model_filename}")
        with open(config_filename, 'w') as file:
            # Write the list of numbers
            file.write("Layer size: " +
                       ', '.join(map(str, model1.layer_sizes)) + '\n')
            # Write the string content
            file.write("Activation: " + model1.activation + '\n')
            file.write("Natoms: " + str(model1.n_atoms) + '\n')
            phi_group_list = model1.phi_group.tolist()
            phi_group_list = model1.phi_group.tolist()
            file.write("Phi Group: " +
                       ', '.join(map(str, phi_group_list)) + '\n')
            psi_group_list = model1.psi_group.tolist()  # Convert tensor to list
            file.write("Psi Group: " +
                       ', '.join(map(str, psi_group_list)) + '\n')
        print(f"Config saved to {config_filename}")


def descriptor_phipsi(x: torch.float32, n_atoms, phi_group, psi_group):
    x_n_d_3 = torch.reshape(x, (x.shape[0], n_atoms, 3))
    x_phi_group = x_n_d_3[:, phi_group, :]
    x_psi_group = x_n_d_3[:, psi_group, :]

    cosphi, sinphi = compute_dihedral_cossin_1(x_phi_group)
    cospsi, sinpsi = compute_dihedral_cossin_1(x_psi_group)

    descriptors = torch.stack((sinphi, cosphi, sinpsi, cospsi), dim=1)
    return descriptors


class NNphipsi_overdamped_1(FunctionModelWithDescriptor):
    def __init__(self, layer_sizes, n_atoms, phi_group,
                 psi_group, activation='sigmoid'):
        super(
            NNphipsi_overdamped_1,
            self).__init__(
            layer_sizes,
            activation=activation,
            using_descriptor=True)
        self.phi_group = phi_group
        self.psi_group = psi_group
        self.n_atoms = n_atoms

    def descriptor(self, x_and_v: torch.float32):
        x_and_v_n_d_3 = torch.reshape(
            x_and_v, (x_and_v.shape[0], self.n_atoms, 3))
        x = x_and_v_n_d_3
        x_phi_group = x[:, self.phi_group, :]
        x_psi_group = x[:, self.psi_group, :]

        cosphi, sinphi = compute_dihedral_cossin_1(x_phi_group)
        cospsi, sinpsi = compute_dihedral_cossin_1(x_psi_group)

        # print(sinphi.shape,cosphi.shape,sinpsi.shape,cospsi.shape,dsinphidt.shape,dcosphidt.shape,dsinpsidt.shape,dcospsidt.shape)
        descriptors = torch.stack((sinphi, cosphi, sinpsi, cospsi), dim=1)
        # print(descriptors)
        return descriptors

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def save(self, examples, filename):

        model1 = self.to(torch.device('cpu'))
        examples = examples.to(torch.device('cpu')).detach()
        traced_script_module = torch.jit.trace(
            model1.forward, example_inputs=examples)
        traced_script_module.save(filename)


class NNd2(FunctionModelWithDescriptor):
    def __init__(self, layer_sizes, n_atoms, activation='sigmoid'):
        super(
            NNd2,
            self).__init__(
            layer_sizes,
            activation=activation,
            using_descriptor=True)
        self.phi_group = phi_group
        self.psi_group = psi_group
        self.n_atoms = n_atoms

    def descriptor(self, x_and_v: torch.float32):
        x_and_v_n_d_3 = torch.reshape(
            x_and_v, (x_and_v.shape[0], self.n_atoms * 2, 3))
        x, v = x_and_v_n_d_3[:, :self.n_atoms,
                             :], x_and_v_n_d_3[:, self.n_atoms:, :]
        dx = x - x[:, 0, :].unsqueeze(1)
        dv = v - v[:, 0, :].unsqueeze(1)
        dx2 = torch.sum(dx**2, dim=2, keepdim=False)
        dx2dt = torch.sum(2 * dx * dv, dim=2, keepdim=False)

        descriptors = torch.cat((dx2, dx2dt), dim=1)
        # print(descriptors)
        return descriptors


class NNd2_45(FunctionModelWithDescriptor):
    def __init__(self, layer_sizes, n_atoms, activation='sigmoid'):
        super(
            NNd2_45,
            self).__init__(
            layer_sizes,
            activation=activation,
            using_descriptor=True)
        self.phi_group = phi_group
        self.psi_group = psi_group
        self.n_atoms = n_atoms
        pairs = list(
            itertools.combinations(
                range(10),
                2))  # Generates 45 pairs
        self.pairs = torch.tensor(pairs)  # Convert to Tensor of shape (45, 2)

        # Compute pairwise distances
        self.i_indices = self.pairs[:, 0]  # First indices of the pairs
        self.j_indices = self.pairs[:, 1]

    def descriptor(self, x_and_v: torch.float32):
        x_and_v_n_d_3 = torch.reshape(
            x_and_v, (x_and_v.shape[0], self.n_atoms * 2, 3))
        x, v = x_and_v_n_d_3[:, :self.n_atoms,
                             :], x_and_v_n_d_3[:, self.n_atoms:, :]
        x_i_group = x[:, self.i_indices, :]
        x_j_group = x[:, self.j_indices, :]
        v_i_group = v[:, self.i_indices, :]
        v_j_group = v[:, self.j_indices, :]
        dx = x_i_group - x_j_group
        dv = v_i_group - v_j_group
        dx2 = torch.sum(dx**2, dim=2, keepdim=False)
        r = torch.sqrt(dx2)
        #dx2dt = torch.sum(2 * dx * dv, dim=2, keepdim=False)
        drdt = torch.sum(dx * dv, dim=2, keepdim=False)/r

        descriptors = torch.cat((r, drdt), dim=1)
        # print(descriptors)
        return descriptors


def preprocessing_data(data: np.ndarray, n_atoms: int):
    data_torch = torch.tensor(data, dtype=torch.float32)
    data_torch_n_d_3 = torch.reshape(
        data_torch, (data_torch.shape[0], n_atoms, 3))
    return data_torch_n_d_3


def read_mass(topology_file):
    # Open the topology file
    with open(topology_file, "r") as file:
        lines = file.readlines()

    # Initialize variables
    mass_info = []
    inside_atoms_block = False

    # Parse the [ atoms ] section
    for line in lines:
        line = line.strip()
        if line.startswith("[ atoms ]"):
            inside_atoms_block = True
            continue
        if inside_atoms_block:
            if line == "" or line.startswith("["):
                # Exit the [ atoms ] block
                inside_atoms_block = False
                continue
            if not line.startswith(";"):  # Ignore comment lines
                tokens = line.split()
                if len(tokens) >= 8:  # Ensure the line has enough columns
                    atom_mass = float(tokens[7])  # Mass is in the 8th column
                    mass_info.append(atom_mass)

    # Output results
    return mass_info


def preprocessing_data_np2torch(
        C7eq_xs, C7eq_vs, C7eq_fs, heavy_atom_indices=heavy_atom_indices):
    heavy_dim_indices = np.stack(
        [(3 * x, 3 * x + 1, 3 * x + 2) for x in heavy_atom_indices], axis=0).reshape(-1)
    C7eq_x_and_v = np.concatenate((C7eq_xs, C7eq_vs), axis=1)
    C7eq_xv_heavy = np.concatenate(
        (C7eq_xs[:, heavy_dim_indices], C7eq_vs[:, heavy_dim_indices]), axis=1)
    C7eq_xv_heavy = torch.from_numpy(C7eq_xv_heavy).float()
    C7eq_x_and_v = torch.from_numpy(C7eq_x_and_v).float()
    C7eq_fs_heavy = torch.from_numpy(C7eq_fs[:, heavy_dim_indices]).float()
    C7eq_fs = torch.from_numpy(C7eq_fs).float()
    return C7eq_x_and_v, C7eq_fs, C7eq_xv_heavy, C7eq_fs_heavy


def save_model_phipsi(model: NNphipsi, model_path, config_path):
    with open(config_path, 'w') as file:
        # Write the list of numbers
        file.write("Layer size: " +
                   ', '.join(map(str, model.layer_sizes)) + '\n')
        # Write the string content
        file.write("Activation: " + model.activation + '\n')
        file.write("Natoms: " + str(model.n_atoms) + '\n')
        phi_group_list = model.phi_group.tolist()
        phi_group_list = model.phi_group.tolist()
        file.write("Phi Group: " + ', '.join(map(str, phi_group_list)) + '\n')
        psi_group_list = model.psi_group.tolist()  # Convert tensor to list
        file.write("Psi Group: " + ', '.join(map(str, psi_group_list)) + '\n')
    print(f"Model saved to {config_path}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def load_model_phipsi(model_path, config_path, device='cpu'):
    print(f"Configuration loaded from {config_path}")
    with open(config_path, 'r') as file:
        for line in file:
            if line.startswith("Layer size:"):
                # Extract the numbers and convert them to a list of integers
                numbers_part = line[len("Layer size: "):].strip()
                layer_sizes = list(map(int, numbers_part.split(', ')))
            elif line.startswith("Activation:"):
                # Extract the string content
                activation = line[len("Activation: "):].strip()
            elif line.startswith("Natoms:"):
                # Extract the string content
                n_atoms = int(line[len("Natoms: "):].strip())
            elif line.startswith("Phi Group:"):
                # Extract the numbers and convert them to a list of integers
                numbers_part = line[len("Phi Group:"):].strip()
                phi_group = torch.tensor(
                    list(map(int, numbers_part.split(', '))), dtype=torch.int32)
            elif line.startswith("Psi Group:"):
                # Extract the numbers and convert them to a list of integers
                numbers_part = line[len("Psi Group:"):].strip()
                psi_group = torch.tensor(
                    list(map(int, numbers_part.split(', '))), dtype=torch.int32)
    model = NNphipsi_overdamped(
        layer_sizes,
        activation=activation,
        n_atoms=n_atoms,
        phi_group=phi_group,
        psi_group=psi_group)
    model1 = torch.jit.load(model_path)
    model.load_state_dict(model1.state_dict())
    print(f"Model loaded from {model_path}")
    return model


def generate_normal_variables(num_v, vdim, output_file, kbt, mass):
    # Parameters

    # Generate normal random variables
    data = torch.normal(0, 1, size=(num_v, vdim)) * np.sqrt(kbt) / \
        torch.sqrt(mass)  # Each value follows N(0, 1)
    data = data.detach().cpu().numpy()
    # Save to file
    np.savetxt(output_file, data)

    print(f"{num_v * vdim} normal variables saved to {output_file}")


def load_v_file(input_file):
    """
    Load a file containing rows of numbers, and save it to another file.

    Parameters:
        input_file (str): Path to the input file to load.
        output_file (str): Path to the output file to save.
    """
    try:
        # Load the data from the input file
        data = np.loadtxt(input_file)

    except Exception as e:
        print(f"An error occurred: {e}")
    return data

# Example usage

def load_data(file_path,positions_filename,velocities_filename,forces_filename,heavy_atom_indices):

#extract_trr_data(trr_file, tpr_file,C7ax_path,positions_filename,velocities_filename,forces_filename)
    C7eq_xs = read_xvg(file_path+positions_filename)
    C7eq_vs = read_xvg(file_path+velocities_filename)
    C7eq_fs = read_xvg(file_path+forces_filename)
    C7eq_xv,C7eq_fs,C7eq_xv_heavy,C7eq_fs_heavy = preprocessing_data_np2torch(C7eq_xs,C7eq_vs,C7eq_fs,heavy_atom_indices)
    return C7eq_xv,C7eq_fs,C7eq_xv_heavy,C7eq_fs_heavy


def hist2d_mean(x, y, values, args=None, mean=True):
    """
    Create a 2D histogram (grid) where each cell contains the mean value of points in that cell.

    Args:
        x (numpy.ndarray): Array of x-coordinates of points.
        y (numpy.ndarray): Array of y-coordinates of points.
        values (numpy.ndarray): Array of values associated with each point.
        args (dict): Optional dictionary of parameters:
            - xbins (int): Number of bins along the x-axis.
            - ybins (int): Number of bins along the y-axis.
            - xmin (float): Minimum x-coordinate for the grid.
            - xmax (float): Maximum x-coordinate for the grid.
            - ymin (float): Minimum y-coordinate for the grid.
            - ymax (float): Maximum y-coordinate for the grid.

    Returns:
        histogram (numpy.ndarray): 2D array of shape (ybins, xbins) containing the mean values.
        x_edges (numpy.ndarray): Bin edges along the x-axis.
        y_edges (numpy.ndarray): Bin edges along the y-axis.
    """
    # Set default arguments if not provided
    if args is None:
        args = {}
    xbins = args.get('xbins', 100)  # Default number of bins for x-axis
    ybins = args.get('ybins', 100)  # Default number of bins for y-axis
    xmin = args.get('xmin', x.min())  # Default to min of x
    xmax = args.get('xmax', x.max())  # Default to max of x
    ymin = args.get('ymin', y.min())  # Default to min of y
    ymax = args.get('ymax', y.max())  # Default to max of y

    # Define the 2D histogram grid
    x_edges = np.linspace(xmin, xmax, xbins + 1)
    y_edges = np.linspace(ymin, ymax, ybins + 1)

    # Initialize grid for sums and counts
    grid_sums = np.zeros((ybins, xbins))
    grid_counts = np.zeros((ybins, xbins))

    # Digitize the points into bins
    x_indices = np.digitize(x, x_edges) - 1  # Convert to 0-based indices
    y_indices = np.digitize(y, y_edges) - 1

    # Accumulate values and counts for each bin
    for i in range(len(values)):
        # Ignore out-of-bound points
        if 0 <= x_indices[i] < xbins and 0 <= y_indices[i] < ybins:
            grid_sums[y_indices[i], x_indices[i]] += values[i]
            grid_counts[y_indices[i], x_indices[i]] += 1

    # Calculate the mean (avoid division by zero)
    if mean:
        with np.errstate(divide='ignore', invalid='ignore'):
            histogram = np.divide(
                grid_sums,
                grid_counts,
                out=np.zeros_like(grid_sums),
                where=grid_counts > 0)
    else:
        histogram = grid_sums
    return histogram, x_edges, y_edges


def read_COLVAR(filename):
    file_path = filename  # Replace with your actual file name

    # Read the file and filter out lines starting with '#!'
    with open(file_path, "r") as file:
        data_lines = [line for line in file if not line.startswith("#")]

    # Convert the remaining lines into a NumPy array
    data = np.loadtxt(data_lines)

    return data


def write_bias_file(filename, C7eq_path, C7ax_path, model, plumed_file):
    file = f'''
# Run the first command
OMP_NUM_THREADS=1 gmx_mpi mdrun -s em_C7ax.tpr -deffnm {C7ax_path}/em -v -plumed {plumed_file}
gmx_mpi grompp -f nvt.mdp -c {C7ax_path}/em.gro -p topol.top -o {C7ax_path}/nvt_C7ax.tpr
OMP_NUM_THREADS=1 gmx_mpi mdrun -s {C7ax_path}/nvt_C7ax.tpr -deffnm {C7ax_path}/trj -v -plumed {plumed_file}
# plumed driver --mf_trr {C7ax_path}/trj.trr --plumed {plumed_file}
# Update the COLVAR path in the plumed file
sed -i 's|{C7ax_path}/COLVAR|{C7eq_path}/COLVAR|g' {plumed_file}

# Run the second command with the updated plumed file
OMP_NUM_THREADS=1 gmx_mpi mdrun -s em_C7eq.tpr -deffnm {C7eq_path}/em -v -plumed {plumed_file}
gmx_mpi grompp -f nvt.mdp -c {C7eq_path}/em.gro -p topol.top -o {C7eq_path}/nvt_C7eq.tpr
OMP_NUM_THREADS=1 gmx_mpi mdrun -s {C7eq_path}/nvt_C7eq.tpr -deffnm {C7eq_path}/trj -v -plumed {plumed_file}
#!/bin/bash

# Function to extract positions, velocities, and forces from a .trr file

# File paths and parameters for the first .trr file
trr_file_1="./{C7ax_path}/trj.trr"  # Path to the .trr file
tpr_file_1="./long_C7ax.tpr"          # Path to the .tpr topology file
working_path_1="./{C7ax_path}/"

# File paths and parameters for the second .trr file
trr_file_2="./{C7eq_path}/trj.trr"  # Path to the .trr file
tpr_file_2="./long_C7eq.tpr"          # Path to the .tpr topology file
working_path_2="./{C7eq_path}/"

# Common output filenames
positions_filename="positions.xvg"
velocities_filename="velocities.xvg"
forces_filename="forces.xvg"
source ./extract_trr_data.sh
# Call the function for the first .trr file
extract_trr_data "$trr_file_1" "$tpr_file_1" "$working_path_1" "$positions_filename" "$velocities_filename" "$forces_filename"

# Call the function for the second .trr file
extract_trr_data "$trr_file_2" "$tpr_file_2" "$working_path_2" "$positions_filename" "$velocities_filename" "$forces_filename"'''
    with open(filename, 'w') as f:
        f.write(file)


def write_mdp_file(filename,
                   gamma,
                   nstep,
                   nstxout=200,
                   nstvout=200,
                   nstenergy=200,
                   nstlog=200,
                   nstfout=200):
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


def write_shooting_file(filename, gro_file, mdp_file,
                        tpr_file, output_name, plumed_file):
    file = f'''
# Run the first command
gmx_mpi grompp -f {mdp_file} -c {gro_file} -p topol.top -o {tpr_file}
OMP_NUM_THREADS=1 gmx_mpi mdrun -s {tpr_file} -deffnm {output_name} -v
plumed driver --mf_trr {output_name}.trr --plumed {plumed_file}
'''
    with open(filename, 'w') as f:
        f.write(file)


def write_plumed_print_file(filename, Colvar_name, stride=100):
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


def write_plumed_file(filename, C7ax_path, model, l=-3):
    file = f'''
    MOLINFO STRUCTURE=./ala2.pdb

LOAD FILE=./pytorch_model_bias.cpp

phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2
theta: TORSION ATOMS=6,5,7,9
sin_phi: MATHEVAL ARG=phi FUNC=sin(x) PERIODIC=NO
cos_phi: MATHEVAL ARG=phi FUNC=cos(x) PERIODIC=NO
sin_theta: MATHEVAL ARG=theta FUNC=sin(x) PERIODIC=NO
cos_theta: MATHEVAL ARG=theta FUNC=cos(x) PERIODIC=NO

d0: DISTANCE ATOMS=2,5
d1: DISTANCE ATOMS=2,6
d2: DISTANCE ATOMS=2,7
d3: DISTANCE ATOMS=2,9
d4: DISTANCE ATOMS=2,11
d5: DISTANCE ATOMS=2,15
d6: DISTANCE ATOMS=2,16
d7: DISTANCE ATOMS=2,17
d8: DISTANCE ATOMS=2,19
d9: DISTANCE ATOMS=5,6
d10: DISTANCE ATOMS=5,7
d11: DISTANCE ATOMS=5,9
d12: DISTANCE ATOMS=5,11
d13: DISTANCE ATOMS=5,15
d14: DISTANCE ATOMS=5,16
d15: DISTANCE ATOMS=5,17
d16: DISTANCE ATOMS=5,19
d17: DISTANCE ATOMS=6,7
d18: DISTANCE ATOMS=6,9
d19: DISTANCE ATOMS=6,11
d20: DISTANCE ATOMS=6,15
d21: DISTANCE ATOMS=6,16
d22: DISTANCE ATOMS=6,17
d23: DISTANCE ATOMS=6,19
d24: DISTANCE ATOMS=7,9
d25: DISTANCE ATOMS=7,11
d26: DISTANCE ATOMS=7,15
d27: DISTANCE ATOMS=7,16
d28: DISTANCE ATOMS=7,17
d29: DISTANCE ATOMS=7,19
d30: DISTANCE ATOMS=9,11
d31: DISTANCE ATOMS=9,15
d32: DISTANCE ATOMS=9,16
d33: DISTANCE ATOMS=9,17
d34: DISTANCE ATOMS=9,19
d35: DISTANCE ATOMS=11,15
d36: DISTANCE ATOMS=11,16
d37: DISTANCE ATOMS=11,17
d38: DISTANCE ATOMS=11,19
d39: DISTANCE ATOMS=15,16
d40: DISTANCE ATOMS=15,17
d41: DISTANCE ATOMS=15,19
d42: DISTANCE ATOMS=16,17
d43: DISTANCE ATOMS=16,19
d44: DISTANCE ATOMS=17,19

q: PYTORCH_MODEL_BIAS FILE={model} ARG=sin_phi,cos_phi,sin_theta,cos_theta ALPHA={l}
potential: BIASVALUE ARG=q.bias-0
PRINT ARG=phi,theta,potential.bias,q.cvforce-0,,q.cvforce-1,q.cvforce-2,q.cvforce-3 FILE={C7ax_path}/COLVAR STRIDE=200'''
    with open(filename, 'w') as f:
        f.write(file)


def get_gamma_data_label(file_path):
    """
    Process a file to extract gammas and data labels into separate lists.

    Parameters:
        file_path (str): Path to the input file.

    Returns:
        tuple: A tuple containing two lists:
            - gammas: A list of integers (from the first column).
            - data_labels: A list of strings (from the second column).
    """
    gammas = []
    data_labels = []

    try:
        # Open and read the file line by line
        with open(file_path, "r") as file:
            for line in file:
                # Split the line into two parts: gamma and data label
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:  # Ensure the line has both parts
                    gamma = parts[0]  # Convert the first part to a float
                    label = parts[1]       # The second part is the label

                    # Append to the respective lists
                    gammas.append(gamma)
                    data_labels.append(label)

        return gammas, data_labels

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return [], []
    except ValueError as e:
        print(f"Error processing file: {e}")
        return [], []


# Example usage
if False:
    # Example usage of the functions
    topology_file = "ala2/simulation/topol.top"
    mass = read_mass(topology_file)
    mass = np.array(mass)
    print(mass)
    print('heavy_atom_mass:', mass[heavy_atom_indices])

    trr_file = "ala2/simulation/long_C7ax_3/trj_3.trr"  # Path to the .trr file
    tpr_file = "ala2/simulation/long_C7ax.tpr"       # Path to the .tpr topology file
    highT_path = "ala2/simulation/800K/"  # Working directory for intermediate files
    long_C7ax_path = "ala2/simulation/long_C7ax_333/"
    long_C7eq_path = "ala2/simulation/long_C7eq_333/"
    C7eq_path = "ala2/simulation/long_C7eq/"
    C7ax_path = "ala2/simulation/long_C7ax/"
    positions_filename = "positions.xvg"
    velocities_filename = "velocities.xvg"
    forces_filename = "forces.xvg"
    # extract_trr_data(trr_file, tpr_file,long_C7ax_path,positions_filename,velocities_filename,forces_filename)
    C7eq_xs = read_xvg(C7eq_path + positions_filename)

    C7eq_vs = read_xvg(C7eq_path + velocities_filename)

    C7eq_fs = read_xvg(C7eq_path + forces_filename)

    C7eq_x_and_v = np.concatenate((C7eq_xs, C7eq_vs), axis=1)
    C7eq_x_and_v = torch.from_numpy(C7eq_x_and_v).float()

    C7eq_xs_n_d_3 = preprocessing_data(C7eq_xs, all_atom_num)
    C7eq_vs_n_d_3 = preprocessing_data(C7eq_vs, all_atom_num)
    C7eq_fs_n_d_3 = preprocessing_data(C7eq_fs, all_atom_num)
    sinphi, cosphi, dihedral_phi = compute_dihedral_cossin(
        C7eq_xs_n_d_3[:, phi_group, :])
    sinpsi, cospsi, dihedral_psi = compute_dihedral_cossin(
        C7eq_xs_n_d_3[:, theta_group, :])
    dihedral_phi = dihedral_phi[:int(dihedral_phi.shape[0] / 3)]
    dihedral_psi = dihedral_psi[:int(dihedral_psi.shape[0] / 3)]

    bins = 300  # Number of bins along each dimension
    hist, x_edges, y_edges = np.histogram2d(
        dihedral_phi.numpy(), dihedral_psi.numpy(), bins=bins, density=True)
    epsilon = 1e-3
    log_hist = np.log(hist + epsilon)
    phi_grid, psi_grid = np.meshgrid(
        0.5 * (x_edges[:-1] + x_edges[1:]),  # Bin centers for phi
        0.5 * (y_edges[:-1] + y_edges[1:])   # Bin centers for psi
    )

    # Plot the 2D histogram
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(
        phi_grid,
        psi_grid,
        log_hist.T,
        levels=20,
        cmap="viridis")
    cbar = plt.colorbar(contour)
    cbar.set_label("Log10 (Density)")

    # Add labels and title
    plt.xlabel("Phi (degrees)")
    plt.ylabel("Psi (degrees)")
    plt.title("2D Histogram of Phi and Psi Angles")
    plt.grid(True)

    # Show the plot
    plt.show()

    C7ax_xs = read_xvg(C7ax_path + positions_filename)
    C7ax_vs = read_xvg(C7ax_path + velocities_filename)
    C7ax_fs = read_xvg(C7ax_path + forces_filename)
    print(C7ax_fs[0, :])
    C7ax_xs_n_d_3 = preprocessing_data(C7ax_xs, all_atom_num)
    C7ax_vs_n_d_3 = preprocessing_data(C7ax_vs, all_atom_num)
    C7ax_fs_n_d_3 = preprocessing_data(C7ax_fs, all_atom_num)
    print(C7ax_xs_n_d_3.shape, C7ax_xs_n_d_3[0, :, :])
    sinphi, cosphi, dihedral_phi1 = compute_dihedral_cossin(
        C7ax_xs_n_d_3[:, phi_group, :])
    sinpsi, cospsi, dihedral_psi1 = compute_dihedral_cossin(
        C7ax_xs_n_d_3[:, theta_group, :])
    # dihedral_phi = dihedral_phi[:int(dihedral_phi.shape[0]/3)]
    # dihedral_psi = dihedral_psi[:int(dihedral_psi.shape[0]/3)]

    # Number of bins along each dimension
    hist, x_edges, y_edges = np.histogram2d(
        torch.cat(
            (dihedral_phi1, dihedral_phi), dim=0).numpy(), torch.cat(
            (dihedral_psi1, dihedral_psi), dim=0).numpy(), bins=bins, density=True)
    log_hist = np.log(hist + epsilon)
    phi_grid, psi_grid = np.meshgrid(
        0.5 * (x_edges[:-1] + x_edges[1:]),  # Bin centers for phi
        0.5 * (y_edges[:-1] + y_edges[1:])   # Bin centers for psi
    )

    # Plot the 2D histogram
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(
        phi_grid,
        psi_grid,
        log_hist.T,
        levels=20,
        cmap="viridis")
    cbar = plt.colorbar(contour)
    cbar.set_label("Log10 (Density)")

    # Add labels and title
    plt.xlabel("Phi (degrees)")
    plt.ylabel("Psi (degrees)")
    plt.title("2D Histogram of Phi and Psi Angles")
    plt.grid(True)

    # Show the plot
    plt.show()

    highT_xs = read_xvg(highT_path + positions_filename)
    highT_vs = read_xvg(highT_path + velocities_filename)
    highT_fs = read_xvg(highT_path + forces_filename)
    print(highT_fs[0, :])
    highT_xs_n_d_3 = preprocessing_data(highT_xs, all_atom_num)
    highT_vs_n_d_3 = preprocessing_data(highT_vs, all_atom_num)
    highT_fs_n_d_3 = preprocessing_data(highT_fs, all_atom_num)
    print(highT_xs_n_d_3.shape, highT_xs_n_d_3[0, :, :])
    sinphi, cosphi, dihedral_phi = compute_dihedral_cossin(
        highT_xs_n_d_3[:, phi_group, :])
    sinpsi, cospsi, dihedral_psi = compute_dihedral_cossin(
        highT_xs_n_d_3[:, theta_group, :])

    # bins = 100  # Number of bins along each dimension
    hist, x_edges, y_edges = np.histogram2d(
        dihedral_phi.numpy(), dihedral_psi.numpy(), bins=bins, density=True)
    log_hist = np.log(hist + epsilon)
    phi_grid, psi_grid = np.meshgrid(
        0.5 * (x_edges[:-1] + x_edges[1:]),  # Bin centers for phi
        0.5 * (y_edges[:-1] + y_edges[1:])   # Bin centers for psi
    )

    # Plot the 2D histogram
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(
        phi_grid,
        psi_grid,
        log_hist.T,
        levels=20,
        cmap="viridis")
    cbar = plt.colorbar(contour)
    cbar.set_label("Log10 (Density)")

    # Add labels and title
    plt.xlabel("Phi (degrees)")
    plt.ylabel("Psi (degrees)")
    plt.title("2D Histogram of Phi and Psi Angles")
    plt.grid(True)

    # Show the plot
    plt.show()

    long_xs = read_xvg(long_C7ax_path + positions_filename)
    # long_vs = read_xvg(long_C7ax_path+velocities_filename)
    # long_fs = read_xvg(long_C7ax_path+forces_filename)
    # print(long_fs[0,:])
    long_xs_n_d_3 = preprocessing_data(long_xs, all_atom_num)
    # long_vs_n_d_3 = preprocessing_data(long_vs, all_atom_num)
    # long_fs_n_d_3 = preprocessing_data(long_fs, all_atom_num)
    print(long_xs_n_d_3.shape, long_xs_n_d_3[0, :, :])
    sinphi, cosphi, dihedral_phi = compute_dihedral_cossin(
        long_xs_n_d_3[:, phi_group, :])
    sinpsi, cospsi, dihedral_psi = compute_dihedral_cossin(
        long_xs_n_d_3[:, theta_group, :])

    # bins = 100  # Number of bins along each dimension
    hist, x_edges, y_edges = np.histogram2d(
        dihedral_phi.numpy(), dihedral_psi.numpy(), bins=bins, density=True)
    log_hist = np.log(hist + epsilon)
    phi_grid, psi_grid = np.meshgrid(
        0.5 * (x_edges[:-1] + x_edges[1:]),  # Bin centers for phi
        0.5 * (y_edges[:-1] + y_edges[1:])   # Bin centers for psi
    )

    # Plot the 2D histogram
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(
        phi_grid,
        psi_grid,
        log_hist.T,
        levels=20,
        cmap="viridis")
    cbar = plt.colorbar(contour)
    cbar.set_label("Log10 (Density)")

    # Add labels and title
    plt.xlabel("Phi (degrees)")
    plt.ylabel("Psi (degrees)")
    plt.title("2D Histogram of Phi and Psi Angles")
    plt.grid(True)

    # Show the plot
    plt.show()

    # extract_trr_data(trr_file, tpr_file,working_path)


if False:
    kbt = 300 * 0.0083144621
    filename_vs = f"ala2/model/vs_{kbt}.txt"
    mass = np.array(read_mass("ala2/simulation/topol.top"))[heavy_atom_indices]
    mass = torch.tensor(mass, dtype=torch.float32)
    mass = torch.repeat_interleave(mass, 3).unsqueeze(0)
    num_v = 25
    vdim = heavy_atom_indices.shape[0] * 3
    generate_normal_variables(num_v, vdim, filename_vs, kbt, mass)

if False:
    step = 10
    deg_to_rad = np.pi / 180.0
    phi_angles = np.arange(-180, 180, step)
    psi_angles = np.arange(-180, 180, step)
    grid = [(phi_deg, psi_deg, phi_deg * deg_to_rad, psi_deg * deg_to_rad)
            for phi_deg in phi_angles for psi_deg in psi_angles]

    # Run in parallel

    for g in grid:
        restrained_simulation(g)

    '''
    for g in grid:
        phi_deg, psi_deg, _, _ = g
        dir_name = f"phi_{phi_deg}_psi_{psi_deg}"
        plot_colvar(
            f"{dir_name}/COLVAR_restrained",
            f"{dir_name}/COLVAR_restrained.png")
    '''

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
