import torch as tr
import numpy as np
import sys

import ufl
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
import basix


sys.path.insert(1, '/home/lewin/01_promotion/03_Programms/dissertation_code')
from src.neural_networks.neural_network_classes import *
from src.neural_networks.network_training import *
from src.pde_modules.pde_descriptions.elliptic_pde.linear_elliptic_pde import *
# device
device = tr.device("cpu")
# dtype
dtype   = tr.double
npdtype = 'double'

# import the pde 
# pde
pde    = pde_general_elliptic_parShape(dtype=dtype, device=device)
mu_set = pde.parameter_set

# import the trained nn
save_path = './10_trained_networks/nn_model_parametric.pt'
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
# load parameter
nn_model.load_state_dict(tr.load(save_path))
nn_model.eval()


# nn function
def nn_function(x, mu):
    x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=tr.double), 0, 1)

    ones_aux = mu*tr.ones(x_tr.shape[0], 1, dtype=tr.double)
    # evaluate the nn
    value_tr = tr.zeros(x_tr.shape[0], 1, dtype=tr.double)
    value_tr = nn_model(tr.cat((x_tr, ones_aux), dim=1))
    value_np = tr.transpose(value_tr,0,1).detach().numpy()

    return value_np

# get the data structures for the error
sol_norm_H10_disc  = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
dual_norm          = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
coerc_const        = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
conti_const        = tr.zeros(1, mu_set.shape[0], dtype=tr.double)

error_H10_disc     = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
rel_error_H10_disc = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
est_error_H10_ub   = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
rel_est_error_H10  = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
est_error_H10_lb   = tr.zeros(1, mu_set.shape[0], dtype=tr.double)
effectivity        = tr.zeros(1, mu_set.shape[0], dtype=tr.double)


# start loop
name_num = [0, 12, 25, 37, 50, 62, 75, 87, 100]
for mu_idx, num in enumerate(name_num):
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
    el_b  = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    Vconv = functionspace(msh, el_b)

    # Defining the trial and test function
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Interpolate the data functions into finite element spaces
    # Defining the source term
    f_source = fem.Function(V)
    f_source.interpolate(pde.f_fenicsx)
    # Defining the diffusion term
    A_diff = fem.Function(Vdiff)
    A_diff.interpolate(pde.A_fenicsx)
    # Defining the convection term
    b_conv = fem.Function(Vconv)
    b_conv.interpolate(pde.b_fenicsx)
    # Defining the reacion term
    c_reac = fem.Function(V)
    c_reac.interpolate(pde.c_fenicsx)

    # defining the nn term 
    nn_func = fem.Function(V)
    nn_func.interpolate(lambda x: nn_function(x, mu_set[mu_idx]))

    # discrete solution which gets read in
    u_disc_sol = fem.Function(V)

    # read in the values
    # load path for values
    load_path = './11_simulation_data/fem_sol_'+repr(num)+'.npy'
    # load data for discrete solution
    fem_sol_data = np.load(load_path)
    # assign values
    u_disc_sol.x.array.real[:] = fem_sol_data[:]

    # boundary function
    uD = fem.Function(V)
    uD.interpolate(pde.uD_fenicsx)
    
    # Create facet to cell connectivity required to determine boundary facets
    boundary_dofs   = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc              = fem.dirichletbc(uD, boundary_dofs)
    
    # bilinear form for riesz problem is not parameter dependent
    a_riesz_H10 =  (ufl.dot(ufl.grad(u), ufl.grad(v)))*ufl.dx

    # linear form
    residual_NN = ( -(   ufl.dot(ufl.dot(ufl.grad(nn_func), A_diff), ufl.grad(v))\
                       + ufl.dot(b_conv, ufl.grad(nn_func))*v \
                       + c_reac*nn_func*v)\
                   + f_source*v)*ufl.dx


    # define the problem
    prob_riesz_H10 = LinearProblem(a_riesz_H10, residual_NN, bcs=[bc], \
                             petsc_options={"ksp_type": "cg", \
                                           "ksp_rtol": 1e-6, \
                                           "ksp_atol": 1e-10, \
                                           "ksp_max_it": 5000})

    # solve
    riesz_repr = prob_riesz_H10.solve()

    # calculate the norm of the solution
    sol_norm_disc = fem.form(ufl.inner(ufl.grad(u_disc_sol), \
                                       ufl.grad(u_disc_sol)) * ufl.dx)
    sol_norm_local_disc          = fem.assemble_scalar(sol_norm_disc)
    sol_norm_H10_disc[0, mu_idx] = np.sqrt(msh.comm.allreduce(\
                                                         sol_norm_local_disc, \
                                                         op=MPI.SUM))
 
    # calculating the error
    H10_error_disc = fem.form(ufl.inner(ufl.grad(u_disc_sol - nn_func), \
                                        ufl.grad(u_disc_sol - nn_func)) * ufl.dx)
    error_local_disc          = fem.assemble_scalar(H10_error_disc)
    error_H10_disc[0, mu_idx] = np.sqrt(msh.comm.allreduce(error_local_disc, \
                                                           op=MPI.SUM))

    # calculating the relative error
    rel_error_H10_disc[0, mu_idx] = \
                          error_H10_disc[0, mu_idx]/sol_norm_H10_disc[0, mu_idx]

    # estimating the error: dual norm
    norm_dual  = fem.form(ufl.inner(ufl.grad(riesz_repr), \
                                    ufl.grad(riesz_repr))*ufl.dx)
    norm_local = fem.assemble_scalar(norm_dual)
    dual_norm[0, mu_idx]  = np.sqrt(msh.comm.allreduce(norm_local, op=MPI.SUM))
 
    # get the coercivity constant
    coerc_const[0, mu_idx] = pde.coercivity_const(mu_set[mu_idx])
    conti_const[0, mu_idx] = pde.continuity_const(mu_set[mu_idx])

    # calc the estimated error
    est_error_H10_ub[0, mu_idx]     = dual_norm[0, mu_idx]/coerc_const[0, mu_idx]
    rel_est_error_H10[0, mu_idx] = dual_norm[0, mu_idx]/(coerc_const[0, mu_idx]\
                                  *sol_norm_H10_disc[0, mu_idx])

    # lower bound
    est_error_H10_lb[0, mu_idx]     = dual_norm[0, mu_idx]/conti_const[0, mu_idx]

    # calc the effectivity
    effectivity[0, mu_idx] = est_error_H10_ub[0, mu_idx]/error_H10_disc[0, mu_idx]



# Only print the error on one process
if msh.comm.rank == 0:
    print(f"parameter_num   H10_norm_sol    dual_norm_residual   coercivity  continuity   abs_H10_error   rel_H10_error   abs_est_error_ub    rel_est_error  abs_est_error_lb  effectivity")


if msh.comm.rank == 0:
    for i, _ in enumerate(mu_set):
        print(f"{i:8d}       {sol_norm_H10_disc[0,i].data:7.8f}         {dual_norm[0,i].data:7.8f}         {coerc_const[0,i].data:7.8f}  {conti_const[0,i].data:7.8f}  {error_H10_disc[0,i].data:7.8f}    {rel_error_H10_disc[0,i].data:7.8f}      {est_error_H10_ub[0,i].data:7.8f}       {rel_est_error_H10[0,i].data:7.8f}   {est_error_H10_lb[0,i].data:7.8f}   {effectivity[0,i].data:7.8f}")



