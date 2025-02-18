import timeit
import torch as tr
import numpy as np
import sys
from math import pi

from mpi4py import MPI
import ufl
from dolfinx.fem.petsc import LinearProblem
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
from dolfinx import plot
from dolfinx import default_scalar_type
import basix
import basix.ufl
from basix import CellType, ElementFamily, LagrangeVariant, QuadratureType
import pyvista


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

# define discretization
N_x = 1
N_y = 1
# polynomial degree
poly_degree = 12

# general cell type
cell_type = CellType.quadrilateral


# create the mesh
msh = mesh.create_rectangle(MPI.COMM_WORLD, \
                            [[0, 0], [4, 0.5-1e-6]], \
                            [N_x, N_y], \
                            cell_type=mesh.CellType.quadrilateral)

# initialize the bdry facets
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)

# spectral elements
elem_V = basix.ufl.element(ElementFamily.P, 
                           cell_type, 
                           poly_degree, 
                           lagrange_variant=LagrangeVariant.gll_isaac)
elem_Vdiff = basix.ufl.element(ElementFamily.P, 
                               cell_type, 
                               poly_degree, 
                               lagrange_variant=LagrangeVariant.gll_isaac, 
                               shape=(2,2))

# function spaces
V     = functionspace(msh, elem_V)
Vdiff = functionspace(msh, elem_Vdiff)

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

# boundary function
uD = fem.Function(V)
uD.interpolate(pde.uD_fenicsx)

# Create facet to cell connectivity required to determine boundary facets
boundary_dofs   = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc              = fem.dirichletbc(uD, boundary_dofs)

################################ create measure ################################
# create the quadrature rule
pts, wts = basix.make_quadrature(cell_type, 
                                 2*poly_degree-1, 
                                 rule=QuadratureType(2))

metadata = {"quadrature_rule": "custom",
            "quadrature_points": pts, "quadrature_weights": wts}
dx_meas = ufl.dx(domain=msh, metadata=metadata)


# bilinear form for riesz problem is not parameter dependent
a_riesz_H10 =  (ufl.dot(ufl.grad(u), ufl.grad(v)))*dx_meas

# get the data structures for the error
dual_norm          = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
dual_norm_omega    = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
conti_const        = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
est_error_H10      = tr.zeros(1, mu_set.shape[0], dtype=tr.double)

for mu_idx, mu in enumerate(mu_set):
    # set parameter in diffusion matrix
    A_diff.interpolate(lambda x: pde.A_fenicsx(x,mu))
    # interpolate the nn
    nn_func.interpolate(lambda x: nn_function(x, mu))

    # linear form
    residual_NN = ( -( ufl.dot(ufl.dot(ufl.grad(nn_func), A_diff), \
                               ufl.grad(v)))   \
                    + f_source*v)*dx_meas

    # define the problem
    prob_riesz_H10 = LinearProblem(a_riesz_H10, residual_NN, bcs=[bc], \
            petsc_options={"ksp_type": "preonly", "pc_type": "cholesky"})

    # solve
    solve_times = timeit.timeit("prob_riesz_H10.solve()", 
                                setup="from __main__ import prob_riesz_H10",
                                number=10)


# Only print the time on one process
if msh.comm.rank == 0:
    print(f"#### Solving Time   = {solve_times/10 :8.7f}")
    ndofs = V.tabulate_dof_coordinates().shape[0]
    print(f"#### Number of DOFs = {ndofs :8d}")
 



