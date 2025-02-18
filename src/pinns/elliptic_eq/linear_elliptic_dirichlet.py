from ..pinn_bases.cl_pinn_base import *
import torch as tr
import numpy as np



###### Classical PINN
### general elliptic equation with dirichlet boundary conditions
class PINN_elliptic_dirichlet_nD(clPINN_base_nD):
    def __init__(self,
                 nn_model,
                 pde,
                 domain_trainset,
                 bdry_trainset=None,
                 tau_bdry=None,
                 mu_trainset=None,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        super().__init__(nn_model,
                         pde,
                         domain_trainset,
                         bdry_trainset=bdry_trainset,
                         tau_bdry=tau_bdry,
                         mu_trainset=mu_trainset,
                         dtype=dtype,
                         device=device)

        # if not parametric, then eval the data functions a-priori
        if not self.parametric:
            ###### Data-function
            # Right-hand side: shape = (N_dom, 1)
            self.f_rhs       = self.pde.f(self.X)
            # Diffusion matrix: shape = (N_dom, dim, dim)
            self.A_diff      = self.pde.A(self.X)
            # Divergence of Diffusion matrix: shape = (N_dom, dim)
            self.div_A_diff  = self.pde.Div_A(self.X)
            # Convection vector: shape = (N_dom, dim)
            self.b_conv      = self.pde.b(self.X)
            # Reaction function: shape = (N_dom, 1)
            self.c_reac      = self.pde.c(self.X)
            # boundary function values
            if self.X_bdry is not None:
                self.uD          = self.pde.uD(self.X_bdry)

            # check if the pde has an analytic solution
            sol_method = getattr(self.pde, "u_sol", None)
            if callable(sol_method):
                self.u_sol = self.pde.u_sol(self.X)
            else:
                self.u_sol = False

        else:
            # auxilary data structures (frequently used during training)
            self.aux_ones_dom = tr.ones(self.N_domain, \
                                        self.mu_trainset[0].shape[0], \
                                        device=device, \
                                        dtype=dtype)
            if self.X_bdry is not None:
                self.aux_ones_bdry = tr.ones(self.N_bdry, \
                                             self.mu_trainset[0].shape[0], \
                                             device=device, \
                                             dtype=dtype)

            # check if the pde has an analytic solution
            if callable(getattr(self.pde, "u_sol", None)):
                self.u_sol = tr.zeros(self.N_domain, \
                                      self.mu_trainset.shape[0], \
                                      dtype=self.dtype, \
                                      device=self.device)
                for i, mu in enumerate(self.mu_trainset):
                    self.u_sol[:,i:i+1] = self.pde.u_sol(self.X, mu_idx=i)

            elif callable(getattr(self.pde, "u_sol_shapeOpt", None)):
                self.u_sol = self.pde.fem_sol_tr
            else:
                self.u_sol = False


    def get_squared_residual_domain(self, mu=1., **kwargs):
        # device and dtype agnostic code
        device = self.device
        dtype  = self.dtype
        
        ###### domain residual
        # add mu for nn eval
        # shape = (N_dom, dim+p)
        if not self.parametric:
            X_NN = self.X
            ###### Data-function
            # Right-hand side: shape = (N_dom, 1)
            f_rhs       = self.f_rhs
            # Diffusion matrix: shape = (N_dom, dim, dim)
            A_diff      = self.A_diff
            # Divergence of Diffusion matrix: shape = (N_dom, dim)
            div_A_diff  = self.div_A_diff
            # Convection vector: shape = (N_dom, dim)
            b_conv      = self.b_conv
            # Reaction function: shape = (N_dom, 1)
            c_reac      = self.c_reac
        else:
            # shape = (N_dom, dim+P)
            X_NN = tr.cat((self.X, tr.mul(mu, self.aux_ones_dom)), dim=1)
            ###### Data-function
            # Right-hand side: shape = (N_dom, 1)
            f_rhs       = self.pde.f(self.X, mu=mu)
            # Diffusion matrix: shape = (N_dom, dim, dim)
            A_diff      = self.pde.A(self.X, mu=mu)
            # Divergence of Diffusion matrix: shape = (N_dom, dim)
            div_A_diff  = self.pde.Div_A(self.X, mu=mu)
            # Convection vector: shape = (N_dom, dim)
            b_conv      = self.pde.b(self.X, mu=mu)
            # Reaction function: shape = (N_dom, 1)
            c_reac      = self.pde.c(self.X, mu=mu)


        ###### derivatives of neural network w.r.t. the input on the domain
        # NN-Output: shape = (N_dom, 1)
        X_NN_in               = X_NN.clone()
        X_NN_in.requires_grad = True
        NN_out                = self.nn_model(X_NN_in)

        # Gradient: shape = (N_dom, dim)
        grad_R_phi_x = tr.autograd.grad(NN_out, \
                                        X_NN_in, \
                                        tr.ones(self.N_domain, \
                                                1,\
                                                dtype=dtype, \
                                                device=device),\
                                        create_graph=True)[0]

        # Hessian: shape = (N_dom, dim, dim)
        Hess_R_phi_x = tr.zeros(self.N_domain, 
                                self.dim, 
                                self.dim,
                                dtype=dtype,
                                device=device)

        for dim_ in range(self.dim):
            # Gradient of first partial derivative: shape = (N_dom, dim)
            DDR_phi_xx = tr.autograd.grad(grad_R_phi_x[:,dim_].unsqueeze(1), \
                                          X_NN_in, \
                                          tr.ones(grad_R_phi_x.shape[0], \
                                                  1, \
                                                  dtype=dtype, \
                                                  device=device), \
                                          create_graph=True)[0]
            # fill Hessian
            Hess_R_phi_x[:, dim_, :] = DDR_phi_xx


        ###### calculate the pointwise residual for the domain
        diff_part_1 = tr.sum(tr.mul(tr.transpose(A_diff, 1, 2), \
                                    Hess_R_phi_x),\
                             dim=(1,2)).unsqueeze(-1)

        diff_part_2 = tr.matmul(div_A_diff.unsqueeze(1), \
                                grad_R_phi_x.unsqueeze(2)).squeeze(-1)

        diff_part   = -(diff_part_1 + diff_part_2)

        conv_part   = tr.matmul(b_conv.unsqueeze(1), \
                                grad_R_phi_x.unsqueeze(2)).squeeze(-1)

        reac_part   = tr.mul(c_reac, NN_out)

        # domain residual: shape = (N_dom, 1)
        residual_dom = tr.square(tr.sub(f_rhs,\
                                        (diff_part+conv_part+reac_part)))

        return residual_dom


    def get_squared_residual_bdry(self, mu=1., **kwargs):
        ###### bdry residual
        # add mu for nn eval
        # shape = (N_dom, dim+p)
        # depends on mu
        if not self.parametric:
            # if it is not parametric the bdry values are expected to be saved
            # a-priori for performance
            X_NN_bdry = self.X_bdry
            uD        = self.uD
        else:

            # shape = (N_dom, dim+Qb+Qf)
            X_NN_bdry = tr.cat((self.X_bdry, \
                                tr.mul(mu, self.aux_ones_bdry)), dim=1)
            # if it is parametric the bdry values are calculated for each mu 
            uD        = self.pde.uD(self.X_bdry, mu=mu)

        # NN boundary values
        NN_out_bdry = self.nn_model(X_NN_bdry)

        # bdry residual: shape = (N_dom, 1)
        residual_bdry = tr.square(NN_out_bdry - uD)

        return residual_bdry


    def get_squared_error(self, mu_idx=0, **kwargs):
        # device and dtype agnostic code
        device = self.device
        dtype  = self.dtype
        
        ###### error
        if not self.parametric:
            # shape = (N_dom, dim)
            X_NN  = self.X
            # shape = (N_dom, 1)
            u_sol = self.u_sol
        else:
            # shape = (N_dom, dim+P)
            X_NN  = tr.cat((self.X, tr.mul(self.mu_trainset[mu_idx], \
                                           self.aux_ones_dom)), dim=1)
            u_sol = self.u_sol[:,mu_idx:mu_idx+1]

        ###### derivatives of neural network w.r.t. the input on the domain
        # NN-Output: shape = (N_dom, 1)
        NN_out = self.nn_model(X_NN)

        # return squared error
        return tr.square(NN_out - u_sol) 


    def get_squared_error_shapeOpt(self, mu_idx=0, **kwargs):
        # device and dtype agnostic code
        device = self.device
        dtype  = self.dtype
        
        ###### error
        # shape = (N_dom, dim+P)
        X_NN  = tr.cat((self.X[mu_idx], tr.mul(self.mu_trainset[mu_idx], \
                                       self.aux_ones_dom)), dim=1)
        u_sol = self.u_sol[mu_idx]

        ###### derivatives of neural network w.r.t. the input on the domain
        # NN-Output: shape = (N_dom, 1)
        NN_out = self.nn_model(X_NN)

        # return squared error
        return tr.square(NN_out - u_sol) 


