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
tr.set_num_threads(12)

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
domain_trainset = []
u_sol_tr        = []

# start loop
#name_num = [0, 12, 25, 37, 50, 62, 75, 87, 100]
name_num_train = [0, 25, 50, 75, 100]
for i, num in enumerate(name_num_train):
    # read in sawtooth mesh
    vertices, cells = np.genfromtxt("01_mesh_data/rectangle"+repr(num)+"_coords.csv", \
                                    delimiter=",", dtype=np.float64), \
                      np.genfromtxt("01_mesh_data/rectangle"+repr(num)+"_elem.csv", \
                                    delimiter=",", dtype=np.int64)
    cells = cells - 1
    domain          = ufl.Mesh(basix.ufl.element("Lagrange", \
                                                 "triangle", \
                                                 1, \
                                                 shape=(2,)))
    msh             = mesh.create_mesh(MPI.COMM_WORLD, cells, vertices, domain)

    # refine mesh
    refinements = 2
    for _ in range(refinements):
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
    V       = functionspace(msh, ("CG", 1))
    fem_sol = fem.Function(V)
    # load the solution for mesh
    load_path = './11_simulation_data/fem_sol_'+repr(num)+'.npy'
    # load data but train only on the half
    fem_sol_data = np.load(load_path)
    # assign values
    fem_sol.x.array.real[:] = fem_sol_data[:]

    # generate random samples of size N_samples
    square_points = tr.rand(4*N_samples, 2, dtype=dtype, device=device)
    eval_idx      = pde.char_func_omega(square_points, mu=mu_set[i,:])>0.5
    sample_points = square_points[eval_idx[:,0],:][:N_samples,:]
    # append to trainset list
    domain_trainset.append(sample_points)

    # eval now the solution
    u_sol_torch = pde.u_sol_shapeOpt(sample_points, fem_sol, msh)

    # assign to list
    u_sol_tr.append(u_sol_torch)


# assign data to pde class
pde.fem_sol_tr = u_sol_tr


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
                              activation_func=tr.nn.functional.tanh)

##### loss function for classical pinn
lossclass = PINN_elliptic_dirichlet_nD(nn_model,
                                       pde,
                                       domain_trainset,
                                       mu_trainset=mu_set,
                                       dtype=dtype,
                                       device=device)


# scipy wrapper for scipy.optimize.minimize
sciwrapper = ScipyWrapper_PINN(lossclass.nn_model.parameters(),
                               lossclass.nn_model,
                               lossclass.eval_error_shapeOpt,
                               dtype=dtype,
                               device=device)


# initial value
x0 = sciwrapper._flat_param()

# synchronize GPU cores
tr.cuda.synchronize()

# start time measurement
start_training = time.time()


# training phase
w = minimize(sciwrapper._eval_function,
             x0,
             method='TNC',
             jac=sciwrapper._eval_gradient,
             options={'eps': 1e-15, 
                      'scale': None, 
                      'offset': None, 
                      'mesg_num': None, 
                      'maxCGit': 250, 
                      'eta': -1, 
                      'stepmx': 150, 
                      'accuracy': 0, 
                      'minfev': 0, 
                      'ftol': 1e-15, 
                      'xtol': 1e-15, 
                      'gtol': 1e-15, 
                      'rescale': -1, 
                      'disp': True, 
                      'finite_diff_rel_step': None, 
                      'maxfun': 10000})

# synchronize GPU cores
tr.cuda.synchronize()

# start time measurement
elapsed_time_training = time.time() - start_training
print(f"##### \n Training Time = {elapsed_time_training:8.7f} \n")


# save the network parameters
#save_path = './10_trained_networks/nn_model_parametric.pt'
#tr.save(nn_model.state_dict(), save_path)



