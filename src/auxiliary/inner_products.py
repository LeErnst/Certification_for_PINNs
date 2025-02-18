import torch as tr
import numpy as np
import math
from math import log2, ceil, factorial, floor, pi, sqrt
import sys

############################ other products           ##########################
def kron_nD(tensor_list):
    if len(tensor_list) == 0:
        raise RuntimeError('Tensor list has zero length.')
    elif len(tensor_list) == 1:
        return tensor_list[0]
    else:
        x = tr.kron(tensor_list[-2], tensor_list[-1])
        new_tensor_list = [*tensor_list[:-2], x]
        return kron_nD(new_tensor_list)


def torch_cartesian_nD(tensor_list):
    # handle the case when tensor_list contains a single element
    return tr.cartesian_prod(*tensor_list).reshape(-1,len(tensor_list))


############################ inner products and norms ##########################


############## L2 inner product 1D
class L2_product_1d():
    def __init__(self, 
                 domain_disc, 
                 F_K,
                 detJF_K,
                 alpha,
                 support_vec,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        self.dtype  = dtype
        self.device = device
        # grid class with elements, coordinates
        self.domain_disc = domain_disc 
        self.el          = domain_disc.el
        # affine transformation onto reference element
        self.F_K         = F_K
        self.detJF_K     = detJF_K
        # number of support vectors (degree of the quadrature rule)
        self.n           = support_vec.shape[0]
        # vectorize qudrature for one element
        # quadrature 1D
        # support vectors for reference element
        self.X_hat       = support_vec.unsqueeze(1).to(dtype=dtype, \
                                                       device=device)
        # the associated weights
        self.Alpha       = alpha.unsqueeze(0).to(dtype=dtype, device=device)

        # coordinates of element defining nodes
        P1 = self.domain_disc.co[self.el[:,0],:]
        P2 = self.domain_disc.co[self.el[:,1],:]

        ## affine transformation: F_k : K_hat -> K
        # absolute value of the determinant of the jacobi matrix of F_k
        self.absdetB = tr.abs(self.detJF_K(P1, P2))

        # transformed coordinates X = F_K(X_hat) in the physical domain
        self.X = self.F_K(self.X_hat, P1, P2)

    def evaluate(self, f, g):

        # L2-scalar product
        L2product = tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, tr.mul(f(self.X),\
                                                               g(self.X)))))

        return L2product 


############## L2 norm 1D
class L2_norm_1d(L2_product_1d):
    def __init__(self, 
                 domain_disc, 
                 F_K,
                 detJF_K,
                 alpha,
                 support_vec,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        # Get the attributes of L2 product
        L2_product_1d.__init__(self,
                               domain_disc, 
                               F_K,
                               detJF_K,
                               alpha,
                               support_vec,
                               dtype=dtype,
                               device=device)

    def evaluate(self, f):

        # L2-norm
        norm = tr.sqrt(tr.sum(tr.mul(self.absdetB,\
                                      tr.matmul(self.Alpha, f(self.X)**2))))

        return norm 



############## L2-product nD
class L2_product_box_nD():
    def __init__(self, 
                 box_disc, 
                 F_K,
                 detJF_K,
                 alpha,
                 support_vec,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        self.dtype  = dtype
        self.device = device
        # grid class with elements, coordinates
        self.box_disc = box_disc 
        # dimension of box
        self.dim      = self.box_disc.dim
        # element coordinates
        self.elem_coords    = self.box_disc.elem_coords

        # affine transformation onto reference element
        self.F_K         = F_K
        self.detJF_K     = detJF_K
        # quadrature 1D
        self.alpha       = alpha
        self.support_vec = support_vec
        # number of support vectors (degree of the quadrature rule)
        self.n           = support_vec.shape[0]
        # vectorize qudrature for one element
        # quadrature nD
        self.X_hat = tr.cartesian_prod(*[self.support_vec]*self.dim) \
                                       .to(dtype=dtype, device=device)

        if self.dim == 1:
            self.X_hat = self.X_hat.unsqueeze(1)

        # the associated weights
        self.Alpha = tr.unsqueeze(kron_nD([self.alpha]*self.dim),dim=0)\
                                  .to(dtype=dtype, device=device)


        # all elements T = [x_i, x_i+1]**dim
        # shape = (numel,dim)
        P1 = self.elem_coords[0,:,:]
        P2 = self.elem_coords[1,:,:]

        ## affine transformation: F_k : K_hat -> K
        # absolute value of the determinant of the jacobi matrix of F_k
        # shape = (numel, 1, 1)
        self.absdetB = tr.abs(self.detJF_K(P1, P2))

        # transformed coordinates X = F_K(X_hat) in the physical domain
        # shape = (numel, n**dim, dim)
        self.X = self.F_K(self.X_hat, P1, P2)

    def evaluate(self, f, g):

        # L2-product
        L2_product = tr.sum(tr.mul(self.absdetB,\
                                   tr.matmul(self.Alpha, tr.mul(f(self.X), \
                                                                   g(self.X)))))

        return L2_product 


############## L2 norm nD
class L2_norm_box_nD(L2_product_box_nD):
    def __init__(self, 
                 domain_disc, 
                 F_K,
                 detJF_K,
                 alpha,
                 support_vec,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        # Get the attributes of L2 product
        L2_product_box_nD.__init__(self,
                                   domain_disc, 
                                   F_K,
                                   detJF_K,
                                   alpha,
                                   support_vec,
                                   dtype=dtype,
                                   device=device)

    def evaluate(self, f):

        # L2-norm
        norm = tr.sqrt(tr.sum(tr.mul(self.absdetB,\
                                      tr.matmul(self.Alpha, f(self.X)**2))))

        return norm 



class H1_product_1d():
    def __init__(self, 
                 domain_disc, 
                 F_K,
                 detJF_K,
                 alpha,
                 support_vec,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        self.dtype  = dtype
        self.device = device
        # grid class with elements, coordinates
        self.domain_disc = domain_disc 
        self.el          = domain_disc.el
        # affine transformation onto reference element
        self.F_K         = F_K
        self.detJF_K     = detJF_K
        # number of support vectors (degree of the quadrature rule)
        self.n           = support_vec.shape[0]
        # vectorize qudrature for one element
        # quadrature 1D
        # support vectors for reference element
        self.X_hat       = support_vec.unsqueeze(1).to(dtype=dtype, \
                                                       device=device)
        # the associated weights
        self.Alpha       = alpha.unsqueeze(0).to(dtype=dtype, device=device)

        # coordinates of element defining nodes
        P1 = self.domain_disc.co[self.el[:,0],:]
        P2 = self.domain_disc.co[self.el[:,1],:]

        ## affine transformation: F_k : K_hat -> K
        # absolute value of the determinant of the jacobi matrix of F_k
        self.absdetB = tr.abs(self.detJF_K(P1, P2))

        # transformed coordinates X = F_K(X_hat) in the physical domain
        self.X = self.F_K(self.X_hat, P1, P2)

    def evaluate(self, f, Df, g, Dg):

        # L2-scalar product: (f,g)_0
        H1product = tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, tr.mul(f(self.X),\
                                                               g(self.X)))))

        # H1-scalar product: (f,g)_1 = (f,g)_0 + (Df,Dg)_0
        H1product = H1product + tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, tr.mul(Df(self.X),\
                                                               Dg(self.X)))))

        return H1product 



############## H1-product nD
class H1_product_box_nD():
    def __init__(self, 
                 box_disc, 
                 F_K,
                 detJF_K,
                 alpha,
                 support_vec,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        self.dtype  = dtype
        self.device = device
        # grid class with elements, coordinates
        self.box_disc = box_disc 
        # dimension of box
        self.dim      = self.box_disc.dim
        # element coordinates
        self.elem_coords    = self.box_disc.elem_coords

        # affine transformation onto reference element
        self.F_K         = F_K
        self.detJF_K     = detJF_K
        # quadrature 1D
        self.alpha       = alpha
        self.support_vec = support_vec
        # number of support vectors (degree of the quadrature rule)
        self.n           = support_vec.shape[0]
        # vectorize qudrature for one element
        # quadrature nD
        self.X_hat = tr.cartesian_prod(*[self.support_vec]*self.dim) \
                                       .to(dtype=dtype, device=device)

        if self.dim == 1:
            self.X_hat = self.X_hat.unsqueeze(1)

        # the associated weights
        self.Alpha = tr.unsqueeze(kron_nD([self.alpha]*self.dim),dim=0)\
                                  .to(dtype=dtype, device=device)


        # all elements T = [x_i, x_i+1]**dim
        # shape = (numel,dim)
        P1 = self.elem_coords[0,:,:]
        P2 = self.elem_coords[1,:,:]

        ## affine transformation: F_k : K_hat -> K
        # absolute value of the determinant of the jacobi matrix of F_k
        # shape = (numel, 1, 1)
        self.absdetB = tr.abs(self.detJF_K(P1, P2))

        # transformed coordinates X = F_K(X_hat) in the physical domain
        # shape = (numel, n**dim, dim)
        self.X = self.F_K(self.X_hat, P1, P2)

    def evaluate(self, f, Df, g, Dg):

        # L2-scalar product: (f,g)_0
        H1product = tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, tr.mul(f(self.X),\
                                                               g(self.X)))))

        # H1-scalar product: (f,g)_1 = (f,g)_0 + (Df,Dg)_0
        H1product = H1product + tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, \
                                           tr.matmul(Df(self.X).unsqueeze(-2),\
                                                     Dg(self.X).unsqueeze(-1))\
                                                                 .squeeze(-1))))

        return H1product 


############## L2 norm nD
class H1_norm_box_nD(H1_product_box_nD):
    def __init__(self, 
                 domain_disc, 
                 F_K,
                 detJF_K,
                 alpha,
                 support_vec,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        # Get the attributes of H1 product
        H1_product_box_nD.__init__(self,
                                   domain_disc, 
                                   F_K,
                                   detJF_K,
                                   alpha,
                                   support_vec,
                                   dtype=dtype,
                                   device=device)

    def evaluate(self, f, Df):

        # L2-scalar product: (f,f)_0
        H1norm = tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, tr.mul(f(self.X),\
                                                               f(self.X)))))

        # H1-scalar product: (f,f)_1 = (f,f)_0 + (Df,Df)_0
        H1norm = tr.sqrt(H1norm + tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, \
                                           tr.matmul(Df(self.X).unsqueeze(-2),\
                                                     Df(self.X).unsqueeze(-1))\
                                                                .squeeze(-1)))))

        return H1norm


############## H1-product nD
class H2_product_box_nD():
    def __init__(self, 
                 box_disc, 
                 F_K,
                 detJF_K,
                 alpha,
                 support_vec,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        self.dtype  = dtype
        self.device = device
        # grid class with elements, coordinates
        self.box_disc = box_disc 
        # dimension of box
        self.dim      = self.box_disc.dim
        assert self.dim == 1

        # element coordinates
        self.elem_coords    = self.box_disc.elem_coords

        # affine transformation onto reference element
        self.F_K         = F_K
        self.detJF_K     = detJF_K
        # quadrature 1D
        self.alpha       = alpha
        self.support_vec = support_vec
        # number of support vectors (degree of the quadrature rule)
        self.n           = support_vec.shape[0]
        # vectorize qudrature for one element
        # quadrature nD
        self.X_hat = tr.cartesian_prod(*[self.support_vec]*self.dim) \
                                       .to(dtype=dtype, device=device)

        if self.dim == 1:
            self.X_hat = self.X_hat.unsqueeze(1)

        # the associated weights
        self.Alpha = tr.unsqueeze(kron_nD([self.alpha]*self.dim),dim=0)\
                                  .to(dtype=dtype, device=device)


        # all elements T = [x_i, x_i+1]**dim
        # shape = (numel,dim)
        P1 = self.elem_coords[0,:,:]
        P2 = self.elem_coords[1,:,:]

        ## affine transformation: F_k : K_hat -> K
        # absolute value of the determinant of the jacobi matrix of F_k
        # shape = (numel, 1, 1)
        self.absdetB = tr.abs(self.detJF_K(P1, P2))

        # transformed coordinates X = F_K(X_hat) in the physical domain
        # shape = (numel, n**dim, dim)
        self.X = self.F_K(self.X_hat, P1, P2)

    def evaluate(self, f, Df, DDf, g, Dg, DDg):

        # L2-scalar product: (f,g)_0
        H2product = tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, tr.mul(f(self.X),\
                                                               g(self.X)))))

        # H1-scalar product: (f,g)_1 = (f,g)_0 + (Df,Dg)_0
        H2product = H2product + tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, \
                                           tr.matmul(Df(self.X).unsqueeze(-2),\
                                                     Dg(self.X).unsqueeze(-1))\
                                                                 .squeeze(-1))))

        # H2-scalar product: (f,g)_2 = (f,g)_1 + (DDf,DDg)_0
        H2product = H2product + tr.sum(tr.mul(self.absdetB,\
                                              tr.matmul(self.Alpha, \
                                                        tr.mul(DDf(self.X),\
                                                               DDg(self.X)))))

        return H2product 


############## L2 norm nD
class H2_norm_box_nD(H2_product_box_nD):
    def __init__(self, 
                 domain_disc, 
                 F_K,
                 detJF_K,
                 alpha,
                 support_vec,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        # Get the attributes of H1 product
        H2_product_box_nD.__init__(self,
                                   domain_disc, 
                                   F_K,
                                   detJF_K,
                                   alpha,
                                   support_vec,
                                   dtype=dtype,
                                   device=device)

    def evaluate(self, f, Df, DDf):

        # L2-scalar product: (f,f)_0
        H2norm = tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, tr.mul(f(self.X),\
                                                               f(self.X)))))

        # H1-scalar product: (f,f)_1 = (f,f)_0 + (Df,Df)_0
        H2norm = H2norm + tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, \
                                           tr.matmul(Df(self.X).unsqueeze(-2),\
                                                     Df(self.X).unsqueeze(-1))\
                                                                .squeeze(-1))))
        # H2-scalar product: (f,f)_2 = (f,f)_1 + (DDf,DDf)_0
        H2norm = tr.sqrt(H2norm + tr.sum(tr.mul(self.absdetB,\
                                  tr.matmul(self.Alpha, \
                                           tr.matmul(DDf(self.X).unsqueeze(-2),\
                                                     DDf(self.X).unsqueeze(-1))\
                                                                .squeeze(-1)))))

        return H2norm


