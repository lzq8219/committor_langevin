import sys
import os
import subprocess

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add the parent directory to the system path
sys.path.append(parent_dir)

# Now you can import the function from the parent module
from utils import langevin_plumed, langevin_input


def run(cmds):
    for cmd in cmds:
        print(cmd)
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        process.wait()  # Wait for the command to finish

        output, error = process.communicate()

        print(f"Output for command '{cmd}':")
        print(output.decode())


if __name__ == '__main__':
    original_dir = os.getcwd()
    print('start')
    nsteps = int(5e5)
    COLVAR = 'COLVAR'
    func = '(146.7-200*exp(-1*(x-1)^2+0*(x-1)*(y-0)-10*(y-0)^2)-100*exp(-1*(x-0)^2+0*(x-0)*(y-0.5)-10*(y-0.5)^2)-170*exp(-6.5*(x+0.5)^2+11*(x+0.5)*(y-1.5)-6.5*(y-1.5)^2)+15*exp(0.7*(x+1)^2+0.6*(x+1)*(y-1)+0.7*(y-1)^2))'
    stride = 1
    temps = {1, 5, 10, 15, 20, 25}
    friction = 10
    init_A = '-0.5582, 1.4417'
    init_B = '0.6235, 0.0281'
    file_input_A = 'input_A'
    file_input_B = 'input_B'
    dic_A = './A/'
    dic_B = './B/'
    file_plumed = 'plumed.dat'
    for temp in temps:
        # A
        print(f'running temp {temp}')
        langevin_plumed(
            func,
            stride,
            COLVAR +
            f"_{temp:.2f}",
            dic_A +
            file_plumed)
        langevin_input(nsteps, temp, friction, init_A, dic_A + file_input_A)
        os.chdir(dic_A)
        cmds = ['plumed ves_md_linearexpansion ' + file_input_A]
        run(cmds)
        os.chdir('../')

        # B
        langevin_plumed(
            func,
            stride,
            COLVAR +
            f"_{temp:.2f}",
            dic_B +
            file_plumed)
        langevin_input(nsteps, temp, friction, init_B, dic_B + file_input_B)
        os.chdir(dic_B)
        cmds = ['plumed ves_md_linearexpansion ' + file_input_B]
        run(cmds)
        os.chdir('../')
