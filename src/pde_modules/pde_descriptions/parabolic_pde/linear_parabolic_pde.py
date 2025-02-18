import torch as tr
import numpy as np
from math import pi, sqrt
from dolfinx.geometry import *



# general elliptic test cases
class pde_general_parabolic_arkansas():
    def __init__(self, \
                 coords, \
                 parameter_space,\
                 mesh_fenicsx=None, \
                 fem_sol_fenicsx=None, \
                 fem_sol_data=None,\
                 dtype=tr.double, \
                 device=tr.device('cpu')):
        # dtype and device agnostic code
        self.device = device
        self.dtype  = dtype

        # dimension 
        self.dim = 3

        # parametric
        self.parametric = True
        self.P_space    = parameter_space

        # assign the fenicsx stuff
        self.msh_fnx       = mesh_fenicsx
        self.fem_sol_fnx   = fem_sol_fenicsx
        # shape=(p, fem dimension)
        self.fem_sol_data  = fem_sol_data

        # create the vertices of arkansas
        self.arkansas_vertices = coords


    def A(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim,dim)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        A_diff = tr.zeros(*x_shape_, \
                          self.dim, self.dim, \
                          dtype=self.dtype, \
                          device=self.device)

        A_diff[...,0,0] = 0.1
        A_diff[...,1,1] = 1.

        return A_diff


    def Div_A(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        result = tr.zeros(*x_shape_, \
                          self.dim, \
                          dtype=self.dtype, \
                          device=self.device)

        return result


    def b(self, x, mu=1., **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        b = tr.zeros(*x_shape_, \
                     self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

        b[...,0] = (31-mu)*tr.sin(2.*x[...,1])**2
        b[...,1] = (31-mu)*tr.cos((x[...,0]+1.)**(mu/4.))

        return b


    def c(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        result = tr.zeros(*x_shape_, \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        result[..., 0] =  x[..., 0]*x[..., 1] + 1.

        return result


    def f(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        
        # x shape
        x_shape_ = list(x.shape[:-1])
        # data structure
        result = tr.zeros(*x_shape_, \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        result[..., 0] = 10.

        return result


    def uD(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        result = tr.zeros(*x_shape_, \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        return result


    def u_sol(self, x, mu_idx=0, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # from torch to numpy
        x_np = np.zeros((3, x.shape[0]))
        x_np[:self.dim,:] = x.detach().cpu().transpose(0,1)

        # shape (dim,...)
        u_tr = tr.tensor(self.u_sol_fenicsx(x_np, mu_idx=mu_idx), \
                         dtype=self.dtype, \
                         device=self.device).transpose(0,1)

        return u_tr


    ### Fenicsx methods
    def A_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim*dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:self.dim,:], dtype=self.dtype).transpose(0, 1)

        # call method
        A_tr = self.A(x_tr)

        # reshape and torch into numpy
        A_np = tr.transpose(tr.flatten(A_tr, start_dim = 1), 0, 1).detach()\
                                                                  .numpy()

        return A_np


    def b_fenicsx(self, x, mu=1., **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:self.dim,:], dtype=self.dtype).transpose(0, 1)

        # call method
        b_tr = self.b(x_tr, mu=mu)

        # reshape and torch into numpy
        b_np = tr.transpose(b_tr, 0, 1).detach().numpy()

        return b_np


    def c_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:self.dim,:], dtype=self.dtype).transpose(0, 1)

        # call method
        c_tr = self.c(x_tr)

        # reshape and torch into numpy
        c_np = tr.transpose(c_tr, 0, 1).detach().numpy()

        return c_np


    def f_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:self.dim,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        f_tr = self.f(x_tr)
        f_np = tr.transpose(f_tr,0,1).detach().numpy()

        return f_np


    def uD_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:self.dim,:], dtype=self.dtype).transpose(0, 1)

        # call method
        uD_tr = self.uD(x_tr)

        # reshape and torch into numpy
        uD_np = tr.transpose(uD_tr, 0, 1).detach().numpy()

        return uD_np


    def u_sol_fenicsx(self, x, mu_idx=0, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)

        # set the value for the particular parameter mu
        self.fem_sol_fnx.x.array.real[:] = self.fem_sol_data[:,mu_idx]

        # get x in the right shape and calc the tree
        x_np = np.transpose(x)
        tree = bb_tree(self.msh_fnx, self.msh_fnx.geometry.dim)

        # compute collisions
        cell_candidates = compute_collisions_points(tree, x_np)
        colliding_cells = compute_colliding_cells(self.msh_fnx, \
                                                  cell_candidates, \
                                                  x_np)
        # now we have to loop over the cells 
        actual_cells = []
        for i, _ in enumerate(x_np):
            if len(colliding_cells.links(i)) > 0:
                actual_cells.append(colliding_cells.links(i)[0])
            else:
                ValueError("Point "+repr(i)+" is not inside the domain")
        u_sol_np = np.transpose(self.fem_sol_fnx.eval(x_np, actual_cells))

        return u_sol_np


    def mean_value_potential_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:self.dim,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        mvp_tr = self.mean_value_potential(x_tr)
        mvp_np = tr.transpose(mvp_tr,0,1).detach().numpy()

        return mvp_np


    def inf_sup_const(self, mu):
        # Output: Lower bound of inf sup constant, out.shape = (1,)

        return 1./(sqrt(2)*36)


    def continuity_const(self, mu):
        # Output: Continuity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the continuity constant (maximal eigenvalue of diff. matrix)
        return sqrt(2)*6.


    def mean_value_potential(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        # Implements the approximate distance function to impose dirichlet 
        # bdry condtions exactly. Reference:
        # [Exact imposition of boundary conditions with distance functions in
        #  physics-informed deep neural networks]

        # set the tolerances for the bdry handling
        atol = 1e-8
        rtol = 1e-8
        zeros_tensor = tr.zeros(1,dtype=self.dtype, device=self.device)

        # x shape
        x_shape_ = list(x.shape[:-1])

        # mean value potentials (eq. 14), shape=(x.shape[:-1], 1)
        Phi = tr.zeros(x[...,:2].reshape(-1,2).shape[0], \
                       1, \
                       dtype=self.dtype, \
                       device=self.device)

        # difference vectors for all vertices, shape=(N_vert, N_coords, 2)
        R_i =     self.arkansas_vertices.unsqueeze(1) \
                - x[...,:2].reshape(-1,2).unsqueeze(0)

        # get the norm of the difference vectors, shape=(N_vert, N_coords)
        norm_R_i = tr.linalg.norm(R_i, dim=2)

        # get the scalar product of r_i and r_i+1, shape=(N_vert, N_coords)
        scal_prod_R_i = tr.zeros(norm_R_i.shape, 
                                 dtype=self.dtype, 
                                 device=self.device)
        scal_prod_R_i[:-1,:] = tr.sum(R_i[:-1,:,:]*R_i[1:,:,:], dim=2)
        scal_prod_R_i[-1,:]  = tr.sum(R_i[-1,:,:]*R_i[0,:,:], dim=1)

        # get the determinant of r_i and r_i+1, shape=(N_vert, N_coords)
        det_R_i = tr.zeros(norm_R_i.shape, 
                           dtype=self.dtype, 
                           device=self.device)
        det_R_i[:-1,:] = R_i[:-1,:,0]*R_i[1:,:,1] - R_i[:-1,:,1]*R_i[1:,:,0]
        det_R_i[-1,:]  = R_i[-1,:,0]*R_i[0,:,1]   - R_i[-1,:,1]*R_i[0,:,0]

        # get the product of the norm of r_i and norm of r_i+1
        # , shape=(N_vert, N_spatial)
        prod_R_i = tr.zeros(norm_R_i.shape, 
                            dtype=self.dtype, 
                            device=self.device)
        prod_R_i[:-1,:] = norm_R_i[:-1,:]*norm_R_i[1:,:]
        prod_R_i[-1,:]  = norm_R_i[-1,:]*norm_R_i[0,:]

        # get t_i step by step, shape=(N_vert, N_coords)
        t_i          = prod_R_i + scal_prod_R_i
        idx_stab     = tr.isclose(t_i, zeros_tensor, rtol=rtol, atol=atol)
        t_i          = 1./t_i
        t_i           = det_R_i*t_i
        idx_stab      = tr.logical_or(idx_stab, 
                                      tr.isclose(t_i, 
                                                 zeros_tensor, 
                                                 rtol=rtol, atol=atol))

        # get 1/norm(R_i)
        idx_stab = tr.isclose(norm_R_i, zeros_tensor, rtol=rtol, atol=atol)
        norm_R_i = 1./norm_R_i
        idx_stab = tr.logical_or(idx_stab, 
                                 tr.isclose(norm_R_i, 
                                            zeros_tensor, 
                                            rtol=rtol, atol=atol))

        # get W function, shape=(N_vert, N_coords)
        W = tr.zeros(norm_R_i.shape, 
                     dtype=self.dtype, 
                     device=self.device)

        W[:-1,:]    = (norm_R_i[:-1,:] + norm_R_i[1:,:])*t_i[:-1,:]
        W[-1,:]     = (norm_R_i[-1,:]  + norm_R_i[0,:])*t_i[-1,:]
        W           = tr.sum(W, dim=0)
        idx_stab = tr.logical_or(tr.sum(idx_stab, dim=0), 
                                 tr.isclose(W, 
                                            zeros_tensor, 
                                            rtol=rtol, atol=atol))

        # get the approximate distance function
        Phi = 2./W

        # stabilize
        Phi[idx_stab]      = 0.
        Phi[tr.isnan(Phi)] = 0.

        # time is present make tensor product function
        Phi = Phi.reshape(*x_shape_, 1)*tr.tanh(x[...,2:3])
        return Phi



