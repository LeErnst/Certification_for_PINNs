import torch as tr
import numpy as np
import sys

import ufl
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
from dolfinx import mesh
from dolfinx.fem import functionspace, dirichletbc
from dolfinx.fem.bcs import locate_dofs_geometrical
from dolfinx import fem
import basix
import basix.ufl
from basix import CellType, ElementFamily, LagrangeVariant, QuadratureType


sys.path.insert(1, '/home/ul/ul_student/ul_wno19/01_disscode/')
from src.neural_networks.neural_network_classes import *
from src.neural_networks.network_training import *
from src.pde_modules.pde_descriptions.parabolic_pde.linear_parabolic_pde import *
# set the threads for cpu
tr.set_num_threads(40)
# device
device = tr.device("cpu")
# dtype
dtype   = tr.double
npdtype = 'double'

################################ import the pde ################################
parameter_space = [1., 10.]
nx              = 31
mu_set = tr.linspace(parameter_space[0], \
                     parameter_space[1], \
                     nx,\
                     dtype=dtype,
                     device=device).reshape(-1,1)

# pde
arkansas_bdry = np.genfromtxt("arkansas_bdry.csv", 
                              delimiter=",", 
                              dtype=np.float64)
coords_tr = tr.from_numpy(arkansas_bdry)
pde  = pde_general_parabolic_arkansas(coords_tr, 
                                     parameter_space,
                                     dtype=dtype, device=device)
############################ end import the pde ################################


################################ import the NN #################################
# import the trained nn
save_path = './10_trained_networks/nn_model_arkansas_parametric.pt'
### nn model
N_0          = 4
N_1_d        = 128
N_L          = 1
nn_list      = [N_1_d]*8
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
n_div = 8
def nn_function(x, mu):
    # pytorch tensor of x, x_tr.shape = (n,3)
    x_tr = tr.transpose(tr.tensor(x[:3,:], dtype=tr.double), 0, 1)
    # mu tensor to cat it to x_tr
    ones_aux = mu*tr.ones(x_tr.shape[0], 1, dtype=tr.double)
    # cat it, x_tr.shape = (n,4)
    x_tr = tr.cat((x_tr, ones_aux), dim=1)

    # result of nn eval
    value_tr = tr.zeros(x_tr.shape[0], 1, dtype=tr.double)

    # reduce the memory consumption by filling the result partly
    n_offset = x_tr.shape[0]//n_div
    for i in range(n_div):
        value_tr[i*n_offset:(i+1)*n_offset,:] = \
                nn_model(x_tr[i*n_offset:(i+1)*n_offset,:])

    # get the right shape
    value_np = tr.transpose(value_tr,0,1).detach().numpy()

    return value_np
############################ end import the NN #################################


############################## create the mesh #################################
# define discretization
N_x = 4
N_y = 4
N_z = 4
# polynomial degree
poly_degree = 8

# general cell type
cell_type = CellType.hexahedron

# create the mesh
msh = mesh.create_box(MPI.COMM_WORLD, \
                      [[0.1345, 0, 0], [0.783, 1, 1]], \
                      [N_x, N_y, N_z], \
                      cell_type=mesh.CellType.hexahedron)

# initialize the bdry facets
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
########################## end create the mesh #################################


########################### create function space ##############################
# spectral elements
elem_V = basix.ufl.element(ElementFamily.P, 
                           CellType.hexahedron, 
                           poly_degree, 
                           lagrange_variant=LagrangeVariant.gll_isaac)
elem_Vdiff = basix.ufl.element(ElementFamily.P, 
                               CellType.hexahedron, 
                               poly_degree, 
                               lagrange_variant=LagrangeVariant.gll_isaac, 
                               shape=(3,3))
elem_Vconv = basix.ufl.element(ElementFamily.P, 
                               CellType.hexahedron, 
                               poly_degree, 
                               lagrange_variant=LagrangeVariant.gll_isaac, 
                               shape=(3,))

# function spaces
V     = functionspace(msh, elem_V)
Vdiff = functionspace(msh, elem_Vdiff)
Vconv = functionspace(msh, elem_Vconv)

# Defining the trial and test function
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Interpolate the data functions into finite element spaces
# Defining the source term
f_source = fem.Function(V)
f_source.interpolate(pde.f_fenicsx) # pyright: ignore
# Defining the diffusion term
A_diff = fem.Function(Vdiff)
A_diff.interpolate(pde.A_fenicsx) # pyright: ignore
# Defining the convection term
b_conv = fem.Function(Vconv)
# Defining the reacion term
c_reac = fem.Function(V)
c_reac.interpolate(pde.c_fenicsx) # pyright: ignore
# boundary function
uD = fem.Function(V)
uD.interpolate(pde.uD_fenicsx) # pyright: ignore
# defining the nn term 
nn_func = fem.Function(V)
####################### end create function space ##############################


############################# create bdry conds ################################
# Create facet to cell connectivity required to determine boundary facets
# determine the bdry nodes by geometric information
def boundary_D(x):
    # x direction
    bdry_idx = np.isclose(x[0], 0.1345)
    bdry_idx = np.logical_or(bdry_idx,np.isclose(x[0], 0.783))
    # y direction
    bdry_idx = np.logical_or(bdry_idx,np.isclose(x[1], 0))
    bdry_idx = np.logical_or(bdry_idx,np.isclose(x[1], 1))

    return bdry_idx

# determine free nodes geometrically
boundary_dofs = locate_dofs_geometrical(V, boundary_D)
bc            = dirichletbc(uD, boundary_dofs) # pyright: ignore
######################### end create bdry conds ################################


################################ create measure ################################
# create the quadrature rule
pts, wts = basix.make_quadrature(cell_type, 
                                 2*poly_degree-1, 
                                 rule=QuadratureType(2))

metadata = {"quadrature_rule": "custom",
            "quadrature_points": pts, "quadrature_weights": wts}
dx_meas = ufl.dx(domain=msh, metadata=metadata)
############################ end create measure ################################


# bilinear form for riesz problem is not parameter dependent:
a_riesz_L2_H10 =  (u.dx(0)*v.dx(0) + u.dx(1)*v.dx(1))*dx_meas

# get the data structures for the error
dual_norm          = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
dual_norm_omega    = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
conti_const        = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
est_error_X_lb     = tr.zeros(1, mu_set.shape[0], dtype=tr.double)

for mu_idx, mu in enumerate(mu_set):
    # set parameter in convection
    b_conv.interpolate(lambda x: pde.b_fenicsx(x, mu)) # pyright: ignore
    # interpolate the nn
    nn_func.interpolate(lambda x: nn_function(x, mu)) # pyright: ignore

    ################# calculate riesz representative of residual ###############
    # the residual in the dual space of the test space L_2(I;H10)
    residual_NN = (-(nn_func.dx(2)*v \
                    + ufl.dot(ufl.dot(ufl.grad(nn_func), A_diff), ufl.grad(v)) \
                    + ufl.dot(b_conv, ufl.grad(nn_func))*v \
                    + c_reac*nn_func*v) \
                   + f_source*v)*dx_meas

    # Riesz problem to determine the norm of the residual
    prob_riesz_L2_H10 = LinearProblem(a_riesz_L2_H10, residual_NN, bcs=[bc], \
                            petsc_options={"ksp_type": "cg", \
                                           "ksp_rtol": 1e-6, \
                                           "ksp_atol": 1e-10, \
                                           "ksp_max_it": 5000})
    # solve the Riesz problem
    riesz_repr = prob_riesz_L2_H10.solve()
    ############# end calculate riesz representative of residual ###############


    ###################### calculating estimated error #########################
    # estimating the error: norm L2(I,H10) of riesz repr:
    norm_dual  = fem.form( (riesz_repr.dx(0)*riesz_repr.dx(0) \
                          + riesz_repr.dx(1)*riesz_repr.dx(1))*dx_meas)

    norm_local = fem.assemble_scalar(norm_dual)
    dual_norm[0, mu_idx]  = np.sqrt(msh.comm.allreduce(norm_local, op=MPI.SUM))
    ################## end calculating estimated error #########################
 
    #################### calculating bilinear constants ########################
    # get the inf sup constant
    conti_const[0, mu_idx] = pde.continuity_const(mu)
    ################ end calculating bilinear constants ########################

    #################### calculating estimated error ###########################
    # calc the estimated error
    est_error_X_lb[0, mu_idx] = dual_norm[0, mu_idx]/conti_const[0, mu_idx]
    ################ end calculating estimated error ###########################

# Only print the error on one process
if msh.comm.rank == 0:
    print(f"abs_est_error_box_lb")


if msh.comm.rank == 0:
    for i, _ in enumerate(mu_set):
        print(f"{est_error_X_lb[0,i].data:7.8f}")




