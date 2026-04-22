#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
from torch.utils.data import DataLoader, TensorDataset
from nn import FunctionModel, save_model, load_model
import copy
import matplotlib.pyplot as plt
from model_training import train_resample,pinn_loss,build_rightside, train_mass,train_overdamped
from hist import hist_reweight
from utils import *

import logging

# Configure logging



# In[11]:

gamma_data_label_file = "ala2/bias_gamma_data_label.txt"
gammas, data_labels = get_gamma_data_label(gamma_data_label_file)

for l in [-1,-3,-5,-7,-9]:
    for gamma, data_label in zip(gammas, data_labels):

        filename = f'./biased_gamma{gamma}_{data_label}/bias.sh'
        plumed_file = f'./biased_gamma{gamma}_{data_label}/plumed_q0.dat'
        itr_path = f'./biased_gamma{gamma}_{data_label}'
        long_C7ax_path_itr = f'./biased_gamma{gamma}_{data_label}/long_C7ax'
        long_C7eq_path_itr = f'./biased_gamma{gamma}_{data_label}/long_C7eq'
        distilling_path = f"../model/distilling_gamma{gamma}"
        model_file = f"{distilling_path}/gamma{gamma}_{data_label}.pth"
        os.chdir('ala2/simulation')

        if not os.path.exists(itr_path):
            os.makedirs(itr_path)

        if not os.path.exists(long_C7ax_path_itr):
            os.makedirs(long_C7ax_path_itr)

        if not os.path.exists(long_C7eq_path_itr):
            os.makedirs(long_C7eq_path_itr)
        write_bias_file(filename,long_C7eq_path_itr,long_C7ax_path_itr,model_file,plumed_file)
        write_plumed_file(plumed_file,long_C7ax_path_itr,model_file,l = -5)

        cmd0 = f"chmod +x {filename}"
        cmd1 = f"bash {filename}"
        subprocess.run(cmd0,shell=True)
        subprocess.run(cmd1,shell=True)
        


        os.chdir("../..")
