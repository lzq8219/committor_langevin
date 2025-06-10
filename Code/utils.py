import numpy as np


class basefile:
    def __init__(self) -> None:
        pass

    def set_value(self, dict) -> None:
        for key, value in dict.items():
            setattr(self, key, value)

    def write(self, filename):
        with open(filename, 'w') as file:
            # Get all attributes of the object as a dictionary
            params_dict = vars(self)

        # Write all parameters and their values to the file
            for key, value in params_dict.items():
                file.write(f'{key} = {value}\n')


class mdfile(basefile):
    def __init__(self,
                 integrator='md',
                 nsteps=500000,
                 dt=0.002,
                 nstxout=500,
                 nstvout=-1,
                 nstxtcout=500,
                 nstenergy=-1,
                 nstlog=-1,
                 continuation='yes',
                 constraint_algorithm='lincs',
                 constraints='h-bonds',
                 lincs_iter=1,
                 lincs_order=4,
                 nstlist=100,
                 rlist=1.159,
                 rcoulomb=1.0,
                 rvdw=1.0,
                 coulombtype='PME',
                 pme_order=4,
                 fourierspacing=0.16,
                 tcoupl='V-rescale',
                 tc_grps='Protein Non-Protein',
                 tau_t='0.1 0.1',
                 ref_t='300 300',
                 pcoupl='no',
                 pbc='xyz',
                 DispCorr='EnerPres',
                 gen_vel='no') -> None:
        self.integrator = integrator
        self.nsteps = nsteps
        self.dt = dt
        self.nstxout = nstxout
        self.nstvout = nstvout
        if nstvout == -1:
            self.nstvout = self.nsteps
        self.nstxtcout = nstxtcout
        self.nstenergy = nstenergy
        if nstenergy == -1:
            self.nstenergy = self.nsteps
        self.nstlog = nstlog
        if nstlog == -1:
            self.nstlog = self.nsteps
        self.continuation = continuation
        self.constraint_algorithm = constraint_algorithm
        self.constraints = constraints
        self.lincs_iter = lincs_iter
        self.lincs_order = lincs_order
        self.nstlist = nstlist
        self.rlist = rlist
        self.rcoulomb = rcoulomb
        self.rvdw = rvdw
        self.coulombtype = coulombtype
        self.pme_order = pme_order
        self.fourierspacing = fourierspacing
        self.tcoupl = tcoupl
        self.tc_grps = tc_grps
        self.tau_t = tau_t
        self.ref_t = ref_t
        self.pcoupl = pcoupl
        self.pbc = pbc
        self.DispCorr = DispCorr
        self.gen_vel = gen_vel


class nvtfile(basefile):
    def __init__(self,
                 integrator='md',
                 nsteps=50000,
                 dt=0.002,
                 nstxout=500,
                 nstvout=-1,
                 nstxtcout=500,
                 nstenergy=-1,
                 nstlog=-1,
                 continuation='no',
                 constraint_algorithm='lincs',
                 constraints='h-bonds',
                 lincs_iter=1,
                 lincs_order=4,
                 ns_type='grid',
                 nstlist=5,
                 rlist=1.0,
                 rcoulomb=1.0,
                 rvdw=1.0,
                 coulombtype='PME',
                 pme_order=4,
                 fourierspacing=0.16,
                 tcoupl='V-rescale',
                 tc_grps='Protein Non-Protein',
                 tau_t='0.1 0.1',
                 ref_t='300 300',
                 pcoupl='no',
                 pbc='xyz',
                 DispCorr='EnerPres',
                 gen_vel='yes',
                 gen_temp=300,
                 gen_seed=-1) -> None:
        super().__init__()
        self.integrator = integrator
        self.nsteps = nsteps
        self.dt = dt
        self.nstxout = nstxout
        self.nstvout = nstvout
        if nstvout == -1:
            self.nstvout = self.nsteps
        self.nstxtcout = nstxtcout
        self.nstenergy = nstenergy
        if nstenergy == -1:
            self.nstenergy = self.nsteps
        self.nstlog = nstlog
        if nstlog == -1:
            self.nstlog = self.nsteps
        self.continuation = continuation
        self.constraint_algorithm = constraint_algorithm
        self.constraints = constraints
        self.lincs_iter = lincs_iter
        self.lincs_order = lincs_order
        self.nstlist = nstlist
        self.rlist = rlist
        self.rcoulomb = rcoulomb
        self.rvdw = rvdw
        self.coulombtype = coulombtype
        self.pme_order = pme_order
        self.fourierspacing = fourierspacing
        self.tcoupl = tcoupl
        self.tc_grps = tc_grps
        self.tau_t = tau_t
        self.ref_t = ref_t
        self.pcoupl = pcoupl
        self.pbc = pbc
        self.DispCorr = DispCorr
        self.gen_vel = gen_vel
        self.gen_temp = gen_temp
        self.gen_seed = gen_seed
        self.ns_type = ns_type


class emfile(basefile):
    def __init__(self,
                 integrator='steep',
                 emtol=1000.0,
                 emstep=0.01,
                 nsteps=50000,
                 nstlist=1,
                 ns_type='grid',
                 rlist=1.0,
                 coulombtype='PME',
                 rcoulomb=1.0,
                 rvdw=1.0,
                 pbc='xyz') -> None:
        super().__init__()
        self.integrator = integrator
        self.emtol = emtol
        self.emstep = emstep
        self.nsteps = nsteps
        self.nstlist = nstlist
        self.ns_type = ns_type
        self.rlist = rlist
        self.coulombtype = coulombtype
        self.rcoulomb = rcoulomb
        self.rvdw = rvdw
        self.pbc = pbc


def gmx_pdb2gmx(gmx_name, filename, vars=''):
    return gmx_name + ' pdb2gmx -f ' + filename + ' ' + vars


def gmx_editconf(gmx_name, output_name, config_name, vars=''):
    return gmx_name + ' editconf -o ' + \
        output_name + ' -f ' + config_name + ' ' + vars


def gmx_solvate(gmx_name, box_name, sol_name, top_name,
                water_name='spc216.gro', vars=''):
    return gmx_name + ' solvate -cp ' + box_name + ' -o ' + \
        sol_name + ' -p ' + top_name + ' -cs ' + water_name + ' ' + vars


def gmx_grompp(gmx_name, output_name, mdpname, config_name, vars=''):
    return gmx_name + ' grompp -o ' + output_name + ' -f ' + \
        mdpname + ' -c ' + config_name + ' ' + vars


def gmx_mdrun(gmx_name, tpr_name, output_name, vars=''):
    return gmx_name + ' mdrun -s ' + tpr_name + ' -o ' + output_name + ' ' + vars


def langevin_plumed(func, stride: int, COLVAR, filename):
    content = f"""# vim:ft=plumed
UNITS NATURAL
p: POSITION ATOM=1
ene: CUSTOM ARG=p.x,p.y PERIODIC=NO FUNC={func}
pot: BIASVALUE ARG=ene

lwall: LOWER_WALLS ARG=p.x KAPPA=1000 AT=-1.3
uwall: UPPER_WALLS ARG=p.x KAPPA=1000 AT=+1.0

PRINT STRIDE={stride} ARG=p.x,p.y,ene FILE={COLVAR}
    """
    with open(filename, 'w') as file:
        file.write(content)


def langevin_input(nsteps: int, temp, friction, init, filename):
    content = f"""nstep             {nsteps}
tstep             0.005
temperature       {temp}
friction          {friction}
random_seed       4525
dimension         2
replicas          1
basis_functions_1 BF_POWERS ORDER=2 MINIMUM=-3.0 MAXIMUM=+3.0
basis_functions_2 BF_POWERS ORDER=2 MINIMUM=-3.0 MAXIMUM=+3.0
initial_position   {init}
output_potential        out/potential.data
output_potential_grid   100
output_histogram       out/histogram.data
output_coeffs     out/222.dat"""
    with open(filename, 'w') as file:
        file.write(content)


def read_COLVAR(COLVAR):
    data = np.loadtxt(COLVAR, skiprows=1)
    return data


if __name__ == "__main__":

    data = read_COLVAR('simulation/A/COLVAR_1.00')
    print(data)
