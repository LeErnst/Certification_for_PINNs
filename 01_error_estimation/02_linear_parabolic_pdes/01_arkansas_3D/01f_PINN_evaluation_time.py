import time
import torch as tr
import numpy as np
from scipy.optimize import minimize
import sys

from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
import basix

sys.path.insert(1, '/home/ul/ul_student/ul_wno19/01_disscode/')
from src.pde_modules.pde_descriptions.parabolic_pde.linear_parabolic_pde import *
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


# read in arkansas mesh 3d (x,y,t)
vertices = np.genfromtxt("arkansas_coords_3D.csv", 
                         delimiter=",", 
                         dtype=np.float64)
cells    = np.genfromtxt("arkansas_elem_3D.csv", 
                         delimiter=",", 
                         dtype=np.int64)

cells = cells - 1

domain = mesh.ufl.Mesh(basix.ufl.element("Lagrange", 
                                    "tetrahedron", 
                                    1, 
                                    shape=(3,)))

msh    = mesh.create_mesh(MPI.COMM_WORLD, cells, vertices, domain)


# import the pde 
parameter_space = [1., 10.]
nx              = 31
mu_set = tr.linspace(parameter_space[0], \
                     parameter_space[1], \
                     nx,\
                     dtype=dtype,
                     device=device).reshape(-1,1)

# train only on half of the parameter
mu_set = mu_set[::2]

# initialize the bdry facets
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)

# function spaces
V           = functionspace(msh, ("CG", 1))
fem_sol     = fem.Function(V)
# load the solution
load_path = './11_simulation_data/01_fem_sol_parametric.npy'
# load data but train only on the half
fem_sol_data = np.load(load_path)
# train only on half of parameter
fem_sol_data = fem_sol_data[:,::2]

# pde
arkansas_bdry = np.genfromtxt("arkansas_bdry.csv", 
                              delimiter=",", 
                              dtype=np.float64)
coords_tr = tr.from_numpy(arkansas_bdry).to(device=device)
pde  = pde_general_parabolic_arkansas(coords_tr, 
                                     parameter_space,
                                     mesh_fenicsx=msh,
                                     fem_sol_fenicsx=fem_sol, 
                                     fem_sol_data=fem_sol_data,
                                     dtype=dtype, device=device)

# take the nodes as training points
domain_trainset = tr.from_numpy(vertices).to(device=device)

# frequently used during training
aux_ones_dom = tr.ones(domain_trainset.shape[0], \
                            mu_set[0].shape[0], \
                            device=device, \
                            dtype=dtype)
X_NN = tr.cat((domain_trainset, tr.mul(mu_set[0:1,:], aux_ones_dom)), dim=1)


## nn model
N_0          = 4
N_1_d        = 128
N_L          = 1
nn_list      = [N_1_d]*8
nn_list[0]   = N_0
nn_list[-1]  = N_L

# choose the activation function
nn_model = NN_dirichlet(nn_list, \
                        pde.mean_value_potential, \
                        activation_func=tr.nn.functional.tanh,
                        dtype=dtype,
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


