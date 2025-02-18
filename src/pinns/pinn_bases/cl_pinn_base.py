import torch as tr
import numpy as np


class clPINN_base_nD():
    def __init__(self,
                 nn_model,
                 pde,
                 domain_trainset,
                 bdry_trainset=None,
                 tau_bdry=None,
                 mu_trainset=None,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        # data type
        self.dtype  = dtype
        # device
        self.device = device

        # dimension
        self.dim    = pde.dim

        # pde description
        self.pde    = pde 

        # penalty parameter for bdry term
        self.tau_bdry   = tau_bdry

        # parametric flag
        self.parametric = self.pde.parametric

        # parameter training set if is parametric
        if self.parametric:
            self.mu_trainset = mu_trainset

        # neural network
        self.nn_model   = nn_model.to(dtype=dtype, device=device)

        # discretization aka training points
        self.X      = domain_trainset
        self.X_bdry = bdry_trainset

        # size of discretization
        if isinstance(self.X, list):
            self.N_domain = self.X[0].shape[0]
        else:
            self.N_domain = self.X.shape[0]

        if self.X_bdry is not None:
            self.N_bdry      = self.X_bdry.shape[0]


    def eval_residual(self, **kwargs):
        if self.parametric is True:
            loss = 0.
            for mu in self.mu_trainset:
                # get the residual for one mu
                residual_dom  = self.get_squared_residual_domain(mu=mu)
                residual_bdry = self.get_squared_residual_bdry(mu=mu)

                # calculate the loss
                loss     = loss +               tr.sum(residual_dom ) \
                                + self.tau_bdry*tr.sum(residual_bdry)

        else:
            # get the residual for one mu
            residual_dom  = self.get_squared_residual_domain()
            residual_bdry = self.get_squared_residual_bdry()

            # calculate the loss
            loss =                 tr.sum(residual_dom ) \
                   + self.tau_bdry*tr.sum(residual_bdry)

        return loss


    def eval_residual_without_bdry(self, **kwargs):
        if self.parametric is True:
            loss = 0.
            for mu in self.mu_trainset:
                # get the residual for one mu
                residual_dom  = self.get_squared_residual_domain(mu=mu)

                # calculate the loss
                loss     = loss + tr.sum(residual_dom) 

        else:
            # get the residual for one mu
            residual_dom  = self.get_squared_residual_domain()

            # calculate the loss
            loss = tr.sum(residual_dom)

        return loss


    def eval_error(self, **kwargs):
        if self.parametric is True:
            loss = 0.
            for mu_idx, _ in enumerate(self.mu_trainset):
                # get the residual for one mu
                error_dom  = self.get_squared_error(mu_idx=mu_idx)

                # calculate the loss
                loss       = loss + tr.sum(error_dom) 
        else:
            # get the residual for one mu
            error_dom = self.get_squared_error()

            # calculate the loss
            loss      = tr.sum(error_dom)

        return loss

    def eval_error_shapeOpt(self, **kwargs):
        loss = 0.
        for mu_idx, _ in enumerate(self.mu_trainset):
            # get the residual for one mu
            error_dom  = self.get_squared_error_shapeOpt(mu_idx=mu_idx)

            # calculate the loss
            loss       = loss + tr.sum(error_dom) 

        return loss



    def get_squared_residual_domain(self, mu=1., **kwargs):
        # gets overwritten in the child-classes
        pass

    def get_squared_residual_bdry(self, mu=1., **kwargs):
        # gets overwritten in the child-classes
        pass

    def get_squared_error(self, mu=1., **kwargs):
        # gets overwritten in the child-classes
        pass

    def get_squared_error_shapeOpt(self, mu=1., **kwargs):
        # gets overwritten in the child-classes
        pass



