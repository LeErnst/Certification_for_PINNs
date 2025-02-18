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
import dolfinx.io
from pathlib import Path
import basix.ufl
import dolfinx
import basix


sys.path.insert(1, '/home/ul/ul_student/ul_wno19/01_disscode/')
from src.neural_networks.neural_network_classes import *
from src.neural_networks.network_training import *
from src.pde_modules.pde_descriptions.parabolic_pde.linear_parabolic_pde import *
# device
device = tr.device("cpu")
# dtype
dtype   = tr.double
npdtype = 'double'

# import the pde 
parameter_space = [1., 10.]
nx              = 31
mu_set = tr.linspace(parameter_space[0], \
                     parameter_space[1], \
                     nx,\
                     dtype=dtype,
                     device=device)

# pde
arkansas_bdry = np.genfromtxt("arkansas_bdry.csv", 
                              delimiter=",", 
                              dtype=np.float64)
coords_tr = tr.from_numpy(arkansas_bdry).to(device=device, dtype=dtype)
pde  = pde_general_parabolic_arkansas(coords_tr, 
                                      parameter_space,
                                      dtype=dtype, device=device)


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

# set the dirichlet bdry conditions
bc = dirichletbc(uD, boundary_dofs_to)

# linear form
L_disc_sol = f_source*v*ufl.dx
 
for mu_idx, mu in enumerate(mu_set):
    # set parameter in diffusion matrix
    b_conv.interpolate(lambda x: pde.b_fenicsx(x, mu))

    # define bilinear and linear form to determine the disc. solution and dual norm
    # bilinear form
    a_disc_sol = u.dx(2)*v*ufl.dx + \
                    (ufl.dot(ufl.dot(ufl.grad(u), A_diff), ufl.grad(v)) \
                     + ufl.dot(b_conv, ufl.grad(u))*v \
                     + c_reac*u*v)*ufl.dx
   
    # define the problem
    prob_disc_sol  = LinearProblem(a_disc_sol, L_disc_sol, bcs=[bc], \
                            petsc_options={"ksp_type": "gmres", \
                                           "ksp_rtol": 1e-6, \
                                           "ksp_atol": 1e-10, \
                                           "ksp_max_it": 5000})
    # solve
    u_disc_sol = prob_disc_sol.solve()

    if mu_idx == 0:
        sol_vals = np.zeros((u_disc_sol.x.array.real.shape[0], \
                            mu_set.shape[0]))

    sol_vals[:,mu_idx] = u_disc_sol.x.array.real[:]

# safe the first mu and the last one
u_disc_sol.x.array.real[:] = sol_vals[:,0]

# save the solution values
nppath= f'./11_simulation_data/01_fem_sol_parametric.npy'
np.save(nppath, sol_vals)

u_disc_sol.name = "TemperatureFirst"
results_folder = Path("11_simulation_data")
results_folder.mkdir(exist_ok=True, parents=True)
with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_folder / "temperature_first.bp", [u_disc_sol], engine="BP4") as vtx:
    vtx.write(0.0)

# safe the first mu and the last one
u_disc_sol.x.array.real[:] = sol_vals[:,-1]

# save the solution values
nppath= f'./11_simulation_data/01_fem_sol_parametric.npy'
np.save(nppath, sol_vals)

u_disc_sol.name = "TemperatureLast"
results_folder = Path("11_simulation_data")
results_folder.mkdir(exist_ok=True, parents=True)
with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_folder / "temperature_last.bp", [u_disc_sol], engine="BP4") as vtx:
    vtx.write(0.0)




