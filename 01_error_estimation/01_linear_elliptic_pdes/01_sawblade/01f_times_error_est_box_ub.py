import timeit
import torch as tr
import numpy as np
import scipy
import sys
from math import pi

import ufl
from mpi4py import MPI
import dolfinx
from dolfinx.fem.petsc import LinearProblem
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
from dolfinx import plot
from dolfinx import default_scalar_type
import basix
from basix import ElementFamily, CellType, LagrangeVariant, LatticeType


sys.path.insert(1, '/home/ul/ul_student/ul_wno19/01_disscode/')
from src.neural_networks.neural_network_classes import *
from src.neural_networks.network_training import *
from src.pde_modules.pde_descriptions.elliptic_pde.linear_elliptic_pde import *
# device
device = tr.device("cpu")
# dtype
dtype   = tr.double
npdtype = 'double'



# import the pde
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
mu_set   = tr.cartesian_prod(mu_1_set, mu_2_set)


# pde
pde  = pde_general_elliptic_sawblade(n_sawtooth, \
                                     ampli, \
                                     y_offset, \
                                     parameter_space, \
                                     dtype=dtype, device=device)

# import the trained nn
save_path = './10_trained_networks/nn_model_saw_parametric.pt'
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
                        activation_func=tr.nn.functional.tanh)
# load parameter
nn_model.load_state_dict(tr.load(save_path, map_location='cpu'))
nn_model.eval()

# nn function
def nn_function(x, mu):
    x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=tr.double), 0, 1)

    ones_aux = mu*tr.ones(x_tr.shape[0], 2, dtype=tr.double)
    # evaluate the nn
    value_tr = tr.zeros(x_tr.shape[0], 1, dtype=tr.double)
    value_tr = nn_model(tr.cat((x_tr, ones_aux), dim=1))
    value_np = tr.transpose(value_tr,0,1).detach().numpy()

    return value_np

# create the mesh
msh = mesh.create_rectangle(MPI.COMM_WORLD, \
                            [[0, 0], [4, 1]], \
                            [8, 2], \
                            cell_type=mesh.CellType.triangle)
# refine mesh
refinements = 7
for i in range(refinements):
    # initialize the bdry facets
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)

    # refine
    (msh,_,_) = mesh.refine(msh)

# initialize the bdry facets
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)


# function spaces
V     = functionspace(msh, ("CG", 1))
el_A  = basix.ufl.element("Lagrange", "triangle", 1, shape=(2, 2))
Vdiff = functionspace(msh, el_A)

# Defining the trial and test function
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Interpolate the data functions into finite element spaces
# Defining the source term
f_source = fem.Function(V)
f_source.interpolate(pde.f_fenicsx)
# Defining the diffusion term
A_diff = fem.Function(Vdiff)
# defining the nn term 
nn_func = fem.Function(V)

### characteristic function
char_omega = fem.Function(V)
char_omega.interpolate(pde.char_func_omega_fenicsx)

# boundary function
uD = fem.Function(V)
uD.interpolate(pde.uD_fenicsx)

# Create facet to cell connectivity required to determine boundary facets
boundary_dofs   = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc              = fem.dirichletbc(uD, boundary_dofs)

# bilinear form for riesz problem is not parameter dependent
a_riesz_H10 =  (ufl.dot(ufl.grad(u), ufl.grad(v)))*ufl.dx

# get the data structures for the error
dual_norm          = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
dual_norm_omega    = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
coerc_const        = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
est_error_H10      = tr.zeros(1, mu_set.shape[0], dtype=tr.double)

for mu_idx, mu in enumerate(mu_set[0:1]):
    # set parameter in diffusion matrix
    A_diff.interpolate(lambda x: pde.A_fenicsx(x,mu))
    # interpolate the nn
    nn_func.interpolate(lambda x: nn_function(x, mu))

    # linear form
    residual_NN = ( -( ufl.dot(ufl.dot(ufl.grad(nn_func), A_diff), \
                               ufl.grad(v))*char_omega )   \
                    + f_source*v*char_omega)*ufl.dx

    # define the problem
    prob_riesz_H10 = LinearProblem(a_riesz_H10, residual_NN, bcs=[bc], \
                         petsc_options={"ksp_type": "cg", \
                                        "ksp_rtol": 1e-6, "ksp_atol": 1e-10, \
                                        "ksp_max_it": 5000})
    # solve
    solve_times = timeit.timeit("prob_riesz_H10.solve()", 
                                setup="from __main__ import prob_riesz_H10",
                                number=10)


# Only print the time on one process
if msh.comm.rank == 0:
    print(f"#### Solving Time   = {solve_times/10 :8.7f}")
    ndofs = V.tabulate_dof_coordinates().shape[0]
    print(f"#### Number of DOFs = {40*ndofs :8d}")
    




