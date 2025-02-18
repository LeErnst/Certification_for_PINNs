import time
import torch as tr
import numpy as np
from scipy.optimize import minimize
import sys
from math import floor, ceil

import ufl
from mpi4py import MPI
import dolfinx
from dolfinx.fem.petsc import LinearProblem
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
from dolfinx import plot
import basix


sys.path.insert(1, '/home/ul/ul_student/ul_wno19/01_disscode/')
from src.pde_modules.pde_descriptions.elliptic_pde.linear_elliptic_pde import *
from src.neural_networks.neural_network_classes import *
from src.neural_networks.network_training import *
from src.pinns.elliptic_eq.linear_elliptic_dirichlet  import *


# set the threads for cpu
#tr.set_num_threads(40)

# device
device = tr.device("cuda:0")
#device = tr.device("cpu")
# dtype
dtype   = tr.double
npdtype = 'double'


# data for sawblade
#N = 1024
n_sawtooth  = 8
ampli       = 1
y_offset    = 0.5
parameter_space = [[0.1, 1.], [0.05, 0.1]]
nx              = 7
ny              = 7
mu_1_set = tr.linspace(parameter_space[0][0], \
                       parameter_space[0][1], \
                       nx,\
                       dtype=dtype,
                       device=device)
mu_2_set = tr.linspace(parameter_space[1][0], \
                       parameter_space[1][1], \
                       ny,\
                       dtype=dtype,
                       device=device)

# train set of the parameter
mu_set = tr.cartesian_prod(mu_1_set, mu_2_set)

# read in sawtooth mesh
vertices, cells = np.genfromtxt("sawblade_coords.csv", \
                                delimiter=",", dtype=np.float64), \
                  np.genfromtxt("sawblade_elem.csv", \
                                delimiter=",", dtype=np.int64)
cells           = cells - 1
domain          = ufl.Mesh(basix.ufl.element("Lagrange", \
                                             "triangle", \
                                             1, \
                                             shape=(2,)))
msh             = mesh.create_mesh(MPI.COMM_WORLD, cells, vertices, domain)
# refine mesh
refinements = 0
for i in range(refinements):
    # initialize the bdry facets
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)

    # refine
    msh = mesh.refine(msh)

# initialize the bdry facets
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)

# function spaces
V           = functionspace(msh, ("CG", 1))
fem_sol     = fem.Function(V)
# load the navier stokes solution for the velocity
load_path = './11_simulation_data/01_fem_sol_parametric.npy'
fem_sol_data = np.load(load_path)

# pde 
pde  = pde_general_elliptic_sawblade(n_sawtooth, \
                                     ampli, \
                                     y_offset, \
                                     parameter_space,\
                                     mesh_fenicsx=msh,\
                                     fem_sol_fenicsx=fem_sol, \
                                     fem_sol_data=fem_sol_data,\
                                     dtype=dtype, device=device)


# draw N uniform points from sawblade domain
N_L_shape       = 4*4096
N_square        = N_L_shape + floor(N_L_shape/4)
square_points   = tr.rand(N_square, 2, dtype=dtype, device=device)
square_points[:,0] = tr.zeros(N_square, dtype=dtype, device=device).uniform_(0,4)
y_tooths        = pde.sawblade(square_points[:,0:1])
idx_sawblade =  (square_points[...,1]>0)*(square_points[...,1] < y_tooths[...,0])\
               *(square_points[...,0]>0)*(square_points[...,0] < 4)
domain_trainset = tr.zeros(idx_sawblade[idx_sawblade == 1].shape[0], \
                           2, \
                           dtype=dtype, \
                           device=device)

domain_trainset = square_points[idx_sawblade == 1,:]
# frequently used during training
aux_ones_dom = tr.ones(domain_trainset.shape[0], \
                            mu_set[0].shape[0], \
                            device=device, \
                            dtype=dtype)
X_NN = tr.cat((domain_trainset, tr.mul(mu_set[0:1,:], aux_ones_dom)), dim=1)

### nn model
N_0          = 4
N_1_d        = 64
N_L          = 1
nn_list      = [N_1_d]*6
nn_list[0]   = N_0
nn_list[-1]  = N_L

# choose the activation function
nn_model = NN_dirichlet(nn_list, \
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


