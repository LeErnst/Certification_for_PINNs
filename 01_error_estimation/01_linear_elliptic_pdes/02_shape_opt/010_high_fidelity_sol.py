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
pde  = pde_general_elliptic_parShape(dtype=dtype, device=device)


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

    # boundary function
    uD = fem.Function(V)
    uD.interpolate(pde.uD_fenicsx)
    
    # Create facet to cell connectivity required to determine boundary facets
    boundary_dofs   = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc              = fem.dirichletbc(uD, boundary_dofs)
    
    # define bilinear and linear form to determine the disc. solution and dual norm
    # bilinear form
    a_disc_sol =  (ufl.dot(ufl.dot(ufl.grad(u), A_diff), ufl.grad(v)) \
                 + ufl.dot(b_conv, ufl.grad(u))*v + c_reac*u*v)*ufl.dx


    # linear form
    L_disc_sol = f_source*v*ufl.dx
    
    # define the problem
    prob_disc_sol  = LinearProblem(a_disc_sol, L_disc_sol, bcs=[bc], \
                             petsc_options={"ksp_type": "gmres", \
                                           "ksp_rtol": 1e-6, \
                                           "ksp_atol": 1e-10, \
                                           "ksp_max_it": 5000})
    # solve
    u_disc_sol = prob_disc_sol.solve()
    
    sol_vals = np.zeros((u_disc_sol.x.array.real.shape[0],))
    
    sol_vals[:] = u_disc_sol.x.array.real[:]

    ### save the solution values
    nppath= f"./11_simulation_data/fem_sol_"+repr(num)+".npy"
    np.save(nppath, sol_vals)



