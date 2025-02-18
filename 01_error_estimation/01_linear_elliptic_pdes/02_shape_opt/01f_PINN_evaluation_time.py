import time
import torch as tr
import numpy as np
from scipy.optimize import minimize
import sys
from math import floor, ceil

import ufl
from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
import basix



sys.path.insert(1, '/home/ul/ul_student/ul_wno19/01_disscode/')
from src.pde_modules.pde_descriptions.elliptic_pde.linear_elliptic_pde import *
from src.neural_networks.neural_network_classes import *
from src.neural_networks.network_training import *
from src.pinns.elliptic_eq.linear_elliptic_dirichlet  import *


# set the threads for cpu
#tr.set_num_threads(12)

# device
device = tr.device("cuda:0")
#device = tr.device("cpu")
# dtype
dtype   = tr.double
npdtype = 'double'

# import the pde 
# pde
pde  = pde_general_elliptic_parShape(dtype=dtype, \
                                     device=device)

# parameter set
mu_set = pde.parameter_set[::2,:]

# number of samples for nn training
N_samples       = 2**16

# generate random samples of size N_samples
square_points = tr.rand(4*N_samples, 2, dtype=dtype, device=device)
eval_idx      = pde.char_func_omega(square_points, mu=mu_set[0,:])>0.5
domain_trainset = square_points[eval_idx[:,0],:][:N_samples,:]



# frequently used during training
aux_ones_dom = tr.ones(domain_trainset.shape[0], \
                       mu_set[0].shape[0], \
                       device=device, \
                       dtype=dtype)
X_NN = tr.cat((domain_trainset, tr.mul(mu_set[0:1,:], aux_ones_dom)), dim=1)


### nn model
N_0          = 3
N_1_d        = 48
N_L          = 1
nn_list      = [N_1_d]*6
nn_list[0]   = N_0
nn_list[-1]  = N_L

# choose the activation function
nn_model = NN_dirichlet_param(nn_list, \
                        pde.mean_value_potential, \
                        activation_func=tr.nn.functional.tanh,
                        dtype=tr.double,
                        device=device)


# time measurements
# synchronize GPU cores
tr.cuda.synchronize()

# start time measurement
start_training = time.time()

reps = 1000
for i in range(reps):
    nn_model(X_NN)


# synchronize GPU cores
tr.cuda.synchronize()

# start time measurement
elapsed_time_evaluation = (time.time() - start_training)/reps
print(f"##### \n PINN Evaluation Time = {elapsed_time_evaluation:8.7f} \n")


