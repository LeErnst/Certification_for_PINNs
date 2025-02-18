import torch as tr
import numpy as np
import sys

import ufl
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
from dolfinx import plot
import basix
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
                                     parameter_space,\
                                     dtype=dtype, device=device)


# read in sawtooth mesh
vertices, cells = np.genfromtxt("sawblade_coords.csv", delimiter=",", dtype=np.float64), \
                  np.genfromtxt("sawblade_elem.csv", delimiter=",", dtype=np.int64)
cells = cells - 1
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
# Defining the diffusion term
A_diff = fem.Function(Vdiff)
# Defining the source term
f_source = fem.Function(V)
f_source.interpolate(pde.f_fenicsx)

# check mvp function
mvp = fem.Function(V)
mvp.interpolate(pde.mean_value_potential_fenicsx)

# boundary function
uD = fem.Function(V)
uD.interpolate(pde.uD_fenicsx)

# Create facet to cell connectivity required to determine boundary facets
boundary_dofs   = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc              = fem.dirichletbc(uD, boundary_dofs)

for mu_idx, mu in enumerate(mu_set):
    # set parameter in diffusion matrix
    A_diff.interpolate(lambda x: pde.A_fenicsx(x,mu))

    # define bilinear and linear form to determine the disc. solution and dual norm
    # bilinear form
    a_disc_sol =  (ufl.dot(ufl.dot(ufl.grad(u), A_diff), ufl.grad(v)))*ufl.dx
    
    # linear form
    L_disc_sol = f_source*v*ufl.dx
    
    # define the problem
    prob_disc_sol = LinearProblem(a_disc_sol, L_disc_sol, bcs=[bc], \
                         petsc_options={"ksp_type": "cg", \
                                        "ksp_rtol": 1e-6, "ksp_atol": 1e-10, \
                                        "ksp_max_it": 5000, "pc_type": "ilu"})
    # solve
    u_disc_sol = prob_disc_sol.solve()

    if mu_idx == 0:
        sol_vals = np.zeros((u_disc_sol.x.array.real.shape[0], \
                            mu_set.shape[0]))

    sol_vals[:,mu_idx] = u_disc_sol.x.array.real[:]


### save the solution values
nppath= f'./11_simulation_data/01_fem_sol_parametric.npy'
np.save(nppath, sol_vals)

