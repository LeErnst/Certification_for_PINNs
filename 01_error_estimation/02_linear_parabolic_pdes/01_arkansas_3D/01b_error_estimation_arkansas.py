import torch as tr
import numpy as np
import sys

import ufl
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
from dolfinx import mesh
from dolfinx.fem import functionspace, dirichletbc
from dolfinx.fem.bcs import locate_dofs_topological, locate_dofs_geometrical
from dolfinx import fem
import basix
import basix.ufl


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

# initialize the bdry facets
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
########################## end create the mesh #################################


########################### create function space ##############################
# function spaces
V     = functionspace(msh, ("CG", 1))
el_A  = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3, 3))
Vdiff = functionspace(msh, el_A)
el_b  = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
Vconv = functionspace(msh, el_b)

# Defining the trial and test function
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Interpolate the data functions into finite element spaces
# discrete solution which gets read in
u_disc_sol = fem.Function(V)
# read in the values
# load path for values
load_path = './11_simulation_data/01_fem_sol_parametric.npy'
# load data for discrete solution
fem_sol_data = np.load(load_path)

# Interpolate the data functions into finite element spaces
# Defining the source term
f_source = fem.Function(V)
f_source.interpolate(pde.f_fenicsx)
# Defining the diffusion term
A_diff = fem.Function(Vdiff)
A_diff.interpolate(pde.A_fenicsx)
# Defining the convection term
b_conv = fem.Function(Vconv)
# Defining the reacion term
c_reac = fem.Function(V)
c_reac.interpolate(pde.c_fenicsx)
# boundary function
uD = fem.Function(V)
uD.interpolate(pde.uD_fenicsx)
# defining the nn term 
nn_func = fem.Function(V)
####################### end create function space ##############################


############################# create bdry conds ################################
# Determine the boundary dofs topologically
boundary_dofs_to   = locate_dofs_topological(V, fdim, boundary_facets)

# Load the coordinates of free dofs at final time t=1 
free_coords_final_time = np.genfromtxt("free_coords_final_time.csv", 
                                       delimiter=",", 
                                       dtype=np.double)
 
# determine the node numbers of free nodes at t=1
def boundary_D_finaltime(x):
    x1 = free_coords_final_time[0,0]
    y1 = free_coords_final_time[0,1]
    freenodes_idx = np.logical_and(np.isclose(x[0], x1), np.isclose(x[1], y1))
    for i in range(1,free_coords_final_time.shape[0]):
        x1 = free_coords_final_time[i,0]
        y1 = free_coords_final_time[i,1]
        freenodes_idx = np.logical_or(freenodes_idx,
                                      np.logical_and(np.isclose(x[0], x1), 
                                                     np.isclose(x[1], y1)))
    freenodes_idx = np.logical_and(freenodes_idx, np.isclose(x[2], 1))

    return freenodes_idx

# determine free nodes geometrically
boundary_dofs_free = locate_dofs_geometrical(V, boundary_D_finaltime)

# create boolean vector to delete the free nodes
idx_del = np.ones(boundary_dofs_to.shape, dtype=bool)
# set False for the same nodes
for i in range(idx_del.shape[0]):
    idx_del[i] = np.isin(boundary_dofs_to[i], boundary_dofs_free, invert=True)

# delete the free nodes at final time
boundary_dofs_to = boundary_dofs_to[idx_del]


# delete the nodes at the inital time to get L_2(I;H^1_0(Omega))
# Load the coordinates of free dofs at final time t=1 
free_coords_init_time = np.genfromtxt("free_coords_init_time.csv", 
                                       delimiter=",", 
                                       dtype=np.double)
 
# determine the node numbers of free nodes at t=0
def boundary_D_inittime(x):
    x1 = free_coords_init_time[0,0]
    y1 = free_coords_init_time[0,1]
    freenodes_idx = np.logical_and(np.isclose(x[0], x1), np.isclose(x[1], y1))
    for i in range(1,free_coords_init_time.shape[0]):
        x1 = free_coords_init_time[i,0]
        y1 = free_coords_init_time[i,1]
        freenodes_idx = np.logical_or(freenodes_idx,
                                      np.logical_and(np.isclose(x[0], x1), 
                                                     np.isclose(x[1], y1)))
    freenodes_idx = np.logical_and(freenodes_idx, np.isclose(x[2], 0))

    return freenodes_idx

# determine free nodes geometrically
boundary_dofs_free = locate_dofs_geometrical(V, boundary_D_inittime)

# create boolean vector to delete the free nodes
idx_del = np.ones(boundary_dofs_to.shape, dtype=bool)
# set False for the same nodes
for i in range(idx_del.shape[0]):
    idx_del[i] = np.isin(boundary_dofs_to[i], boundary_dofs_free, invert=True)

# delete the free nodes at initial time
boundary_dofs_to = boundary_dofs_to[idx_del]

# set the dirichlet bdry conditions
bc = dirichletbc(uD, boundary_dofs_to)
######################### end create bdry conds ################################


# bilinear form for riesz problem is not parameter dependent:
a_riesz_L2_H10 =  (u.dx(0)*v.dx(0) + u.dx(1)*v.dx(1))*ufl.dx

# get the data structures for the error
sol_norm_X_disc  = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
dual_norm        = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
infsup_const     = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
conti_const      = tr.zeros(1, mu_set.shape[0], dtype=tr.double)

error_X_disc     = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
rel_error_X_disc = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
est_error_X_ub   = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
est_error_X_lb   = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
rel_est_error_X  = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
effectivity      = tr.zeros(1, mu_set.shape[0], dtype=tr.double)

for mu_idx, mu in enumerate(mu_set):
    # set parameter in diffusion matrix
    b_conv.interpolate(lambda x: pde.b_fenicsx(x, mu))
    # interpolate the nn
    nn_func.interpolate(lambda x: nn_function(x, mu))

 
    ####################### calculate the norm of the solution: ################
    #     || u ||_X^2 = || u_t ||_L_2(I;H-1)^2 + || u ||_L_2(I;H10)^2
    # Part || u_t ||_L_2(I;H-1)^2: Solve Riesz problem:
    #           (w,v)_L2(I;H10) = (u_t,v)_L2(Q), Q := Omega x I

    # assign the values for the solution (pre calculated)
    u_disc_sol.x.array.real[:] = fem_sol_data[:,mu_idx]
    # create the linear form
    time_deriv_form = u_disc_sol.dx(2)*v*ufl.dx

    # create the problem to be solved with pcg as solver
    prob_time_deriv = LinearProblem(a_riesz_L2_H10, time_deriv_form, bcs=[bc], \
                            petsc_options={"ksp_type": "cg", \
                                           "ksp_rtol": 1e-6, \
                                           "ksp_atol": 1e-10, \
                                           "ksp_max_it": 5000})
    # solve the Riesz problem
    w_time_riesz = prob_time_deriv.solve()

    # calculate the norm of the solution in two steps for the 
    # two part of the norm
    # First: the H^-1 part
    sol_norm_disc = fem.form( (w_time_riesz.dx(0)*w_time_riesz.dx(0) \
                             + w_time_riesz.dx(1)*w_time_riesz.dx(1))*ufl.dx)
    sol_norm_local_disc = fem.assemble_scalar(sol_norm_disc)
    sol_norm_X_disc[0, mu_idx] = msh.comm.allreduce(sol_norm_local_disc, \
                                                      op=MPI.SUM)
 
    # Second: the H10 part
    sol_norm_disc = fem.form( (u_disc_sol.dx(0)*u_disc_sol.dx(0) \
                             + u_disc_sol.dx(1)*u_disc_sol.dx(1))*ufl.dx)
    sol_norm_local_disc = fem.assemble_scalar(sol_norm_disc)
    sol_norm_X_disc[0, mu_idx] = np.sqrt(sol_norm_X_disc[0, mu_idx] \
                                    + msh.comm.allreduce(sol_norm_local_disc, \
                                                         op=MPI.SUM))
    ################### end calculate the norm of the solution: ################



    ################ calculating the true error: || uh - NN ||_X ###############
    # Part || uh_t -NN_t ||_L_2(I;H-1)^2: Solve Riesz problem:
    #           (w,v)_L2(I;H10) = (uh_t-NN_t,v)_L2(Q), Q := Omega x I
    time_deriv_form = (u_disc_sol- nn_func).dx(2)*v*ufl.dx

    # create the problem to be solved with pcg as solver
    prob_time_deriv = LinearProblem(a_riesz_L2_H10, time_deriv_form, bcs=[bc], \
                            petsc_options={"ksp_type": "cg", \
                                           "ksp_rtol": 1e-6, \
                                           "ksp_atol": 1e-10, \
                                           "ksp_max_it": 5000})
    # solve the Riesz problem
    w_time_riesz = prob_time_deriv.solve()

    # calculate the X-norm of the error in two steps for the 
    # two parts of the norm
    # First: the H^-1 part
    error_norm_disc = fem.form( (w_time_riesz.dx(0)*w_time_riesz.dx(0) \
                               + w_time_riesz.dx(1)*w_time_riesz.dx(1))*ufl.dx)
    error_norm_local_disc = fem.assemble_scalar(error_norm_disc)
    error_X_disc[0, mu_idx] = msh.comm.allreduce(error_norm_local_disc, \
                                                      op=MPI.SUM)
 
    # Second: the H10 part
    error_norm_disc = fem.form( ((u_disc_sol- nn_func).dx(0)\
                                    *(u_disc_sol- nn_func).dx(0) \
                               + (u_disc_sol- nn_func).dx(1)\
                                    *(u_disc_sol- nn_func).dx(1))*ufl.dx)
    error_norm_local_disc = fem.assemble_scalar(error_norm_disc)
    error_X_disc[0, mu_idx] = np.sqrt(error_X_disc[0, mu_idx] + \
                                    + msh.comm.allreduce(error_norm_local_disc, \
                                                         op=MPI.SUM))
    ############ end calculating the true error: || uh - NN ||_X ###############




    ################ calculating the relative true error #######################
    # calculating the relative error
    rel_error_X_disc[0, mu_idx] = \
                       error_X_disc[0, mu_idx]/sol_norm_X_disc[0, mu_idx]
    ############ end calculating the relative true error #######################


    ################# calculate riesz representative of residual ###############
    # the residual in the dual space of the test space L_2(I;H10)
    residual_NN = (-(nn_func.dx(2)*v \
                    + ufl.dot(ufl.dot(ufl.grad(nn_func), A_diff), ufl.grad(v)) \
                    + ufl.dot(b_conv, ufl.grad(nn_func))*v \
                    + c_reac*nn_func*v) \
                   + f_source*v)*ufl.dx

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
                          + riesz_repr.dx(1)*riesz_repr.dx(1))*ufl.dx)

    norm_local = fem.assemble_scalar(norm_dual)
    dual_norm[0, mu_idx]  = np.sqrt(msh.comm.allreduce(norm_local, op=MPI.SUM))
    ################## end calculating estimated error #########################
 

    #################### calculating bilinear constants ########################
    # get the inf sup constant
    infsup_const[0, mu_idx] = pde.inf_sup_const(mu)
    conti_const[0, mu_idx] = pde.continuity_const(mu)
    ################ end calculating bilinear constants ########################


    #################### calculating estimated error ###########################
    # calc the estimated error
    est_error_X_ub[0, mu_idx] = dual_norm[0, mu_idx]/infsup_const[0, mu_idx]
    est_error_X_lb[0, mu_idx] = dual_norm[0, mu_idx]/conti_const[0, mu_idx]
    ################ end calculating estimated error ###########################


    #################### calculating rel estimated error #######################
    # relative est error
    rel_est_error_X[0, mu_idx] = dual_norm[0, mu_idx]/(infsup_const[0, mu_idx]\
                                  *sol_norm_X_disc[0, mu_idx])
    ################ end calculating rel estimated error #######################


    ######################## calculating effectivity ###########################
    # calc the effectivity:
    effectivity[0, mu_idx] = est_error_X_ub[0, mu_idx]/error_X_disc[0, mu_idx]
    #################### end calculating effectivity ###########################



# Only print the error on one process
if msh.comm.rank == 0:
    print(f"parameter_num   X_norm_sol    dual_norm_residual   infsup    continuity   abs_X_error   rel_X_error   abs_est_error_ub    rel_est_error  abs_est_error_lb  effectivity")


if msh.comm.rank == 0:
    for i, _ in enumerate(mu_set):
        print(f"{i:8d}       {sol_norm_X_disc[0,i].data:7.8f}         {dual_norm[0,i].data:7.8f}         {infsup_const[0,i].data:7.8f}  {conti_const[0,i].data:7.8f}  {error_X_disc[0,i].data:7.8f}    {rel_error_X_disc[0,i].data:7.8f}      {est_error_X_ub[0,i].data:7.8f}       {rel_est_error_X[0,i].data:7.8f}   {est_error_X_lb[0,i].data:7.8f}   {effectivity[0,i].data:7.8f}")


