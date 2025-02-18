import torch as tr
import numpy as np
from math import pi, sqrt
from dolfinx.geometry import *


###### poisson equation
# pde
class pde1_poisson():
    def __init__(self, Qb=5, dtype=tr.double, device=tr.device('cpu')):
        # dtype and device agnostic code
        self.device = device
        self.dtype  = dtype

        self.domain = [(0, 1)]
        self.dim    = 1
        # dimension of parameter space
        self.Qb = Qb
        self.P  = [[1/8, 1]]*self.Qb
        # each parameter is connected to some region of the domain
        self.dom_disc = tr.linspace(self.domain[0][0], \
                                    self.domain[0][1], \
                                    self.Qb + 1, \
                                    dtype=self.dtype, \
                                    device=self.device)


    def a(self, x, mu):
        # Input: x.shape = (N,n,1) , mu.shape = (Qb,)
        # N: number of elements
        # Output: out.shape = (N,1)

        # data structure
        result = tr.zeros(x.shape[0], 1, dtype=self.dtype, device=self.device)

        # number of quadrature points
        n = x.shape[1]
        mid_n = n//2

        for i in range(self.Qb):
            idx =  (x[:,mid_n,:] >= self.dom_disc[i])\
                  *(x[:,mid_n,:] <= self.dom_disc[i+1])
            result[idx] = mu[i]


        return result


    def f(self, x, mu):
        # Input: x.shape = (N,n,dim) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1)

        result = tr.cos(2*pi*x)

        return (4*pi**2)*(result)


# control pde
class pde2_poisson():
    def __init__(self, dtype=tr.double, device=tr.device('cpu')):
        self.Qb = 1
        self.P  = [[1.0, 1.1]]*self.Qb
        self.domain = [(0, 1)]
        self.device = device
        self.dtype  = dtype

    def a(self, x, mu):
        # Input: x.shape = (N,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,1), in this simple case we do not store
        #         constant values, but its broadcastable
        return tr.ones(1, 1, dtype=self.dtype, device=self.device)

    def f(self, x, mu):
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable

        result = tr.zeros(x.shape[0], \
                          x.shape[1], \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        idx = (x[:,:,0] >= 0)*(x[:,:,0] <= 1)
        vals = tr.ones(x.shape, dtype=self.dtype, device=self.device)
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)

        return result

    def F(self, x, mu):
        # Integral of self.f (the one s.t. the bdry conditions are fulfilled)
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable

        # Derivative of the solution 
        return tr.sub(0.5,x)

        # Integral of f: F(x) - F(0), where F(x) = x
        #return x

    def calF(self, x, mu):
        # Integral of self.F (the one s.t. the bdry conditions are fulfilled)
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable

        # The solution 
        return tr.mul(0.5, tr.mul(x, tr.sub(1., x)))

        # Integral of F: calF(x)
        #return 0.5*x**2


    def usol(self, x, mu):
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable
        result = tr.zeros(x.shape[0], \
                          x.shape[1], \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        idx = (x[:,:,0] >= 0)*(x[:,:,0] <= 1)
        vals = tr.mul(0.5, tr.mul(x, tr.sub(1., x)))
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)
 
        return result

    def usol_der(self, x, mu):
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable
        return tr.sub(0.5,x)


# control pde
class pde3_poisson():
    def __init__(self, dtype=tr.double, device=tr.device('cpu')):
        self.Qb = 1
        self.P  = [[1.0, 1.1]]*self.Qb
        self.domain = [(0, 1)]
        self.device = device
        self.dtype  = dtype

    def a(self, x, mu):
        # Input: x.shape = (N,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,1), in this simple case we do not store
        #         constant values, but its broadcastable
        return tr.ones(1, 1, dtype=self.dtype, device=self.device)

    def f(self, x, mu):
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable
        result = tr.zeros(x.shape[0], \
                          x.shape[1], \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        idx = (x[:,:,0] >= 0)*(x[:,:,0] <= 1)
        vals = tr.sin(x)
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)

        return result

    def usol(self, x, mu):
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable

        result = tr.zeros(x.shape[0], \
                          x.shape[1], \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        idx = (x[:,:,0] >= 0)*(x[:,:,0] <= 1)
        vals = tr.sub(tr.sin(x),\
                      tr.mul(x, tr.sin(tr.ones(x.shape, \
                                               dtype=self.dtype, \
                                               device=self.device))))
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)

        return result

    def usol_der(self, x, mu):
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable
        return tr.cos(x)-tr.sin(tr.ones(1, \
                                        dtype=self.dtype, \
                                        device=self.device)) \


# control pde
class pde4_poisson():
    def __init__(self, dtype=tr.double, device=tr.device('cpu')):
        self.Qb = 1
        self.P  = [[1.0, 1.1]]*self.Qb
        self.domain = [(0, 1)]
        self.device = device
        self.dtype  = dtype

    def a(self, x, mu):
        # Input: x.shape = (N,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,1), in this simple case we do not store
        #         constant values, but its broadcastable
        return tr.ones(1, 1, dtype=self.dtype, device=self.device)

    def f(self, x, mu):
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable
        return 4*pi**2*tr.sin(2*pi*x)

    def usol(self, x, mu):
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable
        return tr.sin(2*pi*x)


# control pde
class pde5_poisson():
    def __init__(self, dtype=tr.double, device=tr.device('cpu')):
        self.Qb = 1
        self.P  = [[1.0, 1.1]]*self.Qb
        self.domain = [(0., 1.0)]
        self.device = device
        self.dtype  = dtype

    def a(self, x, mu):
        # Input: x.shape = (N,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,1), in this simple case we do not store
        #         constant values, but its broadcastable
        return tr.ones(1, 1, dtype=self.dtype, device=self.device)

    def f(self, x, mu):
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable

        result = tr.zeros(x.shape[0], \
                          x.shape[1], \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)
        ## 0.3 < x < 0.4
        idx = (x[:,:,0] > 0.3)*(x[:,:,0] <= 0.4)
        vals = 10*x[:,:,0]-3.
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)
        ## 0.4 < x < 0.6
        idx = (x[:,:,0] > 0.4)*(x[:,:,0] <= 0.6)
        vals = 5. - 10*x[:,:,0]
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)
        ## 0.6 < x < 0.7
        idx = (x[:,:,0] > 0.6)*(x[:,:,0] <= 0.7)
        vals = 10*x[:,:,0]-7.
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)

        return result
 

    def usol(self, x, mu):
        # Input: x.shape = (N,n,1) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable

        result = tr.zeros(x.shape[0], \
                          x.shape[1], \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        # u prime
        u_prime = 1/50.

        ## 0 <= x <= 0.3
        idx = (x[:,:,0] >= 0)*(x[:,:,0] <= 0.3)
        vals = u_prime*x[:,:,0]
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)
        ## 0.3 < x <= 0.4
        idx = (x[:,:,0] > 0.3)*(x[:,:,0] <= 0.4)
        vals =    u_prime*x[:,:,0]-5./3.*x[:,:,0]**3+3./2.*x[:,:,0]**2 \
               -  45./100.*x[:,:,0]+9./200
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)
        ## 0.4 < x <= 0.6
        idx = (x[:,:,0] > 0.4)*(x[:,:,0] <= 0.6)
        vals =    u_prime*x[:,:,0]-5./2.*x[:,:,0]**2+5./3.*x[:,:,0]**3 \
                + 23./20.*x[:,:,0]-101./600
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)
        ## 0.6 < x <= 0.7
        idx = (x[:,:,0] > 0.6)*(x[:,:,0] <= 0.7)
        vals =    u_prime*x[:,:,0]-5./3.*x[:,:,0]**3+7./2.*x[:,:,0]**2 \
                - 49./20.*x[:,:,0]+331./600
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)
        ## 0.7 < x <= 1.0
        idx = (x[:,:,0] > 0.7)*(x[:,:,0] <= 1.0)
        vals = u_prime*(x[:,:,0]-1)
        result[idx] = vals[idx].reshape(vals[idx].shape[0],1)
 
        return result



# control pde
class pde6_poisson():
    def __init__(self, dim=2, dtype=tr.double, device=tr.device('cpu')):
        self.Qb = 1
        self.P  = [[1.0, 1.1]]*self.Qb
        self.dim    = dim
        self.domain = [(0, 1)]*self.dim
        self.device = device
        self.dtype  = dtype

    def a(self, x, mu):
        # Input: x.shape = (N,dim) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,1), in this simple case we do not store
        #         constant values, but its broadcastable
        return tr.ones(1, 1, dtype=self.dtype, device=self.device)

    def f(self, x, mu):
        # Input: x.shape = (N,n,dim) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable
        result = tr.ones(1, dtype=self.dtype, device=self.device)
        for dim_ in range(self.dim):
            result = result*tr.cos(2*pi*x[:,:,dim_])
        return ((self.dim*4*pi**2)*(result)).unsqueeze(-1)


    def usol(self, x, mu):
        # Input: x.shape = (N,n,dim) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable
        result = tr.ones(1, dtype=self.dtype, device=self.device)
        for dim_ in range(self.dim):
            result = result*tr.cos(2*pi*x[:,:,dim_])
        return result.unsqueeze(-1)



# control pde
class pde7_poisson():
    def __init__(self, dim=2, dtype=tr.double, device=tr.device('cpu')):
        self.Qb = 1
        self.P  = [[0.1, 1.0]]*self.Qb
        self.dim    = dim
        self.domain = [(0, 1)]*self.dim
        self.device = device
        self.dtype  = dtype

    def a(self, x, mu):
        # Input: x.shape = (N,dim) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,1), in this simple case we do not store
        #         constant values, but its broadcastable
        return tr.ones(1, 1, dtype=self.dtype, device=self.device)

    def f(self, x, mu):
        # Input: x.shape = (N,n,dim) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable
        result = tr.ones(1, dtype=self.dtype, device=self.device)
        for dim_ in range(self.dim):
            result = result*tr.cos(tr.floor(1/mu)*2*pi*x[:,:,dim_])
        return ((self.dim*(2*pi*tr.floor(1/mu))**2)*(result)).unsqueeze(-1)


    def usol(self, x, mu):
        # Input: x.shape = (N,n,dim) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1), in this simple case we do not store
        #         constant values, but its broadcastable
        result = tr.ones(1, dtype=self.dtype, device=self.device)
        for dim_ in range(self.dim):
            result = result*tr.cos(tr.floor(1/mu)*2*pi*x[:,:,dim_])
        return result.unsqueeze(-1)


class pde8_diff_reac():
    def __init__(self, Qb=5, dtype=tr.double, device=tr.device('cpu')):
        ## Currently only 1D
        # dtype and device agnostic code
        self.device = device
        self.dtype  = dtype

        # domain
        self.domain = [(0, 1)]
        self.dim    = 1
        # dimension of parameter space
        self.Qb = Qb
        self.P  = [[1/8, 2]]*self.Qb
        # each parameter is connected to some points within the closed domain
        self.dom_disc = tr.linspace(self.domain[0][0], \
                                    self.domain[0][1], \
                                    self.Qb+2, \
                                    dtype=self.dtype, \
                                    device=self.device)


    def a(self, x, mu):
        # Input: x.shape = (...,dim) , mu.shape = (Qb,)
        # N: number of elements
        # Output: out.shape = (...,1)

        # data structure
        result = tr.zeros(*list(x.shape[:-1]), \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        # first and last mu are set to 1
        x_i   = self.dom_disc[0]
        x_ip1 = self.dom_disc[1]
        idx   =  (x_i <= x)*(x < x_ip1)
        result[idx] =   (mu[0]-1)/(x_ip1-x_i)*x[idx] \
                      + (1*x_ip1 - mu[0]*x_i)/(x_ip1-x_i)
        # last
        x_i   = self.dom_disc[-2]
        x_ip1 = self.dom_disc[-1]
        idx   =  (x_i <= x)*(x <= x_ip1)
        result[idx] =   (1-mu[-1])/(x_ip1-x_i)*x[idx] \
                      + (mu[-1]*x_ip1 - 1*x_i)/(x_ip1-x_i)

        for i in range(1, self.Qb):
            x_i   = self.dom_disc[i]
            x_ip1 = self.dom_disc[i+1]
            idx   =  (x_i <= x)*(x < x_ip1)
            result[idx] =   (mu[i]-mu[i-1])/(x_ip1-x_i)*x[idx] \
                          + (mu[i-1]*x_ip1 - mu[i]*x_i)/(x_ip1-x_i)

        return result


    def grad_a(self, x, mu):
        # Input: x.shape = (...,dim) , mu.shape = (Qb,)
        # N: number of elements
        # Output: out.shape = (...,dim)

        # data structure
        result = tr.zeros(x.shape, dtype=self.dtype, device=self.device)
        # first and last mu are set to 1
        x_i   = self.dom_disc[0]
        x_ip1 = self.dom_disc[1]
        idx   =  (x_i <= x)*(x < x_ip1)
        result[idx] =   (mu[0]-1)/(x_ip1-x_i)
        # last
        x_i   = self.dom_disc[-2]
        x_ip1 = self.dom_disc[-1]
        idx   =  (x_i <= x)*(x <= x_ip1)
        result[idx] =   (1-mu[-1])/(x_ip1-x_i)

        for i in range(1, self.Qb):
            x_i   = self.dom_disc[i]
            x_ip1 = self.dom_disc[i+1]
            idx   =  (x_i <= x)*(x < x_ip1)
            result[idx] =   (mu[i]-mu[i-1])/(x_ip1-x_i)

        return result


    def norm_const_H2(self, mu):
        # Input: mu.shape = (Qb,)
        # N: number of elements
        # Output: Norm constant for c(mu)||v||_2 <= ||B*v||_0, out.shape = (1,)

        a_0 = tr.min(self.a(self.dom_disc.reshape(-1,1), mu))

        return min(a_0, 1)


    def coercivity_const(self, mu):
        # Input: mu.shape = (Qb,)
        # N: number of elements
        # Output: Coercivity constant w.r.t. H1-Norm and reaction coefficient
        #         equals 1, out.shape = (1,)

        return self.norm_const_H2(mu)


    def f(self, x, mu):
        # Input: x.shape = (N,n,dim) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1)

        result = tr.cos(2*pi*x)

        return (4*pi**2)*(result)


# general elliptic test cases
class pde_general_elliptic():
    def __init__(self, dim, dtype=tr.double, device=tr.device('cpu')):
        # dtype and device agnostic code
        self.device = device
        self.dtype  = dtype

        # domain
        self.domain = [(0, 1)]*dim
        self.dim    = 1
        # dimension of parameter space
        self.Qb = Qb
        self.P  = [[1/8, 2]]*self.Qb
        # each parameter is connected to some points within the closed domain
        self.dom_disc = tr.linspace(self.domain[0][0], \
                                    self.domain[0][1], \
                                    self.Qb+2, \
                                    dtype=self.dtype, \
                                    device=self.device)


    def a(self, x, mu):
        # Input: x.shape = (...,dim) , mu.shape = (Qb,)
        # N: number of elements
        # Output: out.shape = (...,1)

        # data structure
        result = tr.zeros(*list(x.shape[:-1]), \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        # first and last mu are set to 1
        x_i   = self.dom_disc[0]
        x_ip1 = self.dom_disc[1]
        idx   =  (x_i <= x)*(x < x_ip1)
        result[idx] =   (mu[0]-1)/(x_ip1-x_i)*x[idx] \
                      + (1*x_ip1 - mu[0]*x_i)/(x_ip1-x_i)
        # last
        x_i   = self.dom_disc[-2]
        x_ip1 = self.dom_disc[-1]
        idx   =  (x_i <= x)*(x <= x_ip1)
        result[idx] =   (1-mu[-1])/(x_ip1-x_i)*x[idx] \
                      + (mu[-1]*x_ip1 - 1*x_i)/(x_ip1-x_i)

        for i in range(1, self.Qb):
            x_i   = self.dom_disc[i]
            x_ip1 = self.dom_disc[i+1]
            idx   =  (x_i <= x)*(x < x_ip1)
            result[idx] =   (mu[i]-mu[i-1])/(x_ip1-x_i)*x[idx] \
                          + (mu[i-1]*x_ip1 - mu[i]*x_i)/(x_ip1-x_i)

        return result


    def grad_a(self, x, mu):
        # Input: x.shape = (...,dim) , mu.shape = (Qb,)
        # N: number of elements
        # Output: out.shape = (...,dim)

        # data structure
        result = tr.zeros(x.shape, dtype=self.dtype, device=self.device)
        # first and last mu are set to 1
        x_i   = self.dom_disc[0]
        x_ip1 = self.dom_disc[1]
        idx   =  (x_i <= x)*(x < x_ip1)
        result[idx] =   (mu[0]-1)/(x_ip1-x_i)
        # last
        x_i   = self.dom_disc[-2]
        x_ip1 = self.dom_disc[-1]
        idx   =  (x_i <= x)*(x <= x_ip1)
        result[idx] =   (1-mu[-1])/(x_ip1-x_i)

        for i in range(1, self.Qb):
            x_i   = self.dom_disc[i]
            x_ip1 = self.dom_disc[i+1]
            idx   =  (x_i <= x)*(x < x_ip1)
            result[idx] =   (mu[i]-mu[i-1])/(x_ip1-x_i)

        return result


    def norm_const_H2(self, mu):
        # Input: mu.shape = (Qb,)
        # N: number of elements
        # Output: Norm constant for c(mu)||v||_2 <= ||B*v||_0, out.shape = (1,)

        a_0 = tr.min(self.a(self.dom_disc.reshape(-1,1), mu))

        return min(a_0, 1)


    def coercivity_const(self, mu):
        # Input: mu.shape = (Qb,)
        # N: number of elements
        # Output: Coercivity constant w.r.t. H1-Norm and reaction coefficient
        #         equals 1, out.shape = (1,)

        return self.norm_const_H2(mu)


    def f(self, x, mu):
        # Input: x.shape = (N,n,dim) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1)

        result = tr.cos(2*pi*x)

        return (4*pi**2)*(result)



# general elliptic test cases
class pde_general_elliptic_v2():
    def __init__(self, dim, dtype=tr.double, device=tr.device('cpu')):
        # dtype and device agnostic code
        self.device = device
        self.dtype  = dtype

        # domain
        self.domain = [(0, 1)]*dim
        self.dim    = 1
        # dimension of parameter space
        self.Qb = Qb
        self.P  = [[1/8, 2]]*self.Qb
        # each parameter is connected to some points within the closed domain
        self.dom_disc = tr.linspace(self.domain[0][0], \
                                    self.domain[0][1], \
                                    self.Qb+2, \
                                    dtype=self.dtype, \
                                    device=self.device)


    def a(self, x, mu):
        # Input: x.shape = (...,dim) , mu.shape = (Qb,)
        # N: number of elements
        # Output: out.shape = (...,1)

        # data structure
        result = tr.zeros(*list(x.shape[:-1]), \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        # first and last mu are set to 1
        x_i   = self.dom_disc[0]
        x_ip1 = self.dom_disc[1]
        idx   =  (x_i <= x)*(x < x_ip1)
        result[idx] =   (mu[0]-1)/(x_ip1-x_i)*x[idx] \
                      + (1*x_ip1 - mu[0]*x_i)/(x_ip1-x_i)
        # last
        x_i   = self.dom_disc[-2]
        x_ip1 = self.dom_disc[-1]
        idx   =  (x_i <= x)*(x <= x_ip1)
        result[idx] =   (1-mu[-1])/(x_ip1-x_i)*x[idx] \
                      + (mu[-1]*x_ip1 - 1*x_i)/(x_ip1-x_i)

        for i in range(1, self.Qb):
            x_i   = self.dom_disc[i]
            x_ip1 = self.dom_disc[i+1]
            idx   =  (x_i <= x)*(x < x_ip1)
            result[idx] =   (mu[i]-mu[i-1])/(x_ip1-x_i)*x[idx] \
                          + (mu[i-1]*x_ip1 - mu[i]*x_i)/(x_ip1-x_i)

        return result


    def grad_a(self, x, mu):
        # Input: x.shape = (...,dim) , mu.shape = (Qb,)
        # N: number of elements
        # Output: out.shape = (...,dim)

        # data structure
        result = tr.zeros(x.shape, dtype=self.dtype, device=self.device)
        # first and last mu are set to 1
        x_i   = self.dom_disc[0]
        x_ip1 = self.dom_disc[1]
        idx   =  (x_i <= x)*(x < x_ip1)
        result[idx] =   (mu[0]-1)/(x_ip1-x_i)
        # last
        x_i   = self.dom_disc[-2]
        x_ip1 = self.dom_disc[-1]
        idx   =  (x_i <= x)*(x <= x_ip1)
        result[idx] =   (1-mu[-1])/(x_ip1-x_i)

        for i in range(1, self.Qb):
            x_i   = self.dom_disc[i]
            x_ip1 = self.dom_disc[i+1]
            idx   =  (x_i <= x)*(x < x_ip1)
            result[idx] =   (mu[i]-mu[i-1])/(x_ip1-x_i)

        return result


    def norm_const_H2(self, mu):
        # Input: mu.shape = (Qb,)
        # N: number of elements
        # Output: Norm constant for c(mu)||v||_2 <= ||B*v||_0, out.shape = (1,)

        a_0 = tr.min(self.a(self.dom_disc.reshape(-1,1), mu))

        return min(a_0, 1)


    def coercivity_const(self, mu):
        # Input: mu.shape = (Qb,)
        # N: number of elements
        # Output: Coercivity constant w.r.t. H1-Norm and reaction coefficient
        #         equals 1, out.shape = (1,)

        return self.norm_const_H2(mu)


    def f(self, x, mu):
        # Input: x.shape = (N,n,dim) , mu.shape = (p,)
        # N: number of elements
        # n: number of support vectors of quadrature rule for 1D
        # Output: out.shape = (N,n,1)

        result = tr.cos(2*pi*x)

        return (4*pi**2)*(result)


# general elliptic test cases
class pde_general_elliptic_Lshape():
    def __init__(self, \
                 mesh_fenicsx=None, \
                 fem_sol_fenicsx=None, \
                 dtype=tr.double, \
                 device=tr.device('cpu')):
        # dtype and device agnostic code
        self.device = device
        self.dtype  = dtype

        # dimension 
        self.dim    = 2

        # parametric
        self.parametric = False

        # assign the fenicsx stuff
        self.msh_fnx       = mesh_fenicsx
        self.fem_sol_fnx   = fem_sol_fenicsx

        # vertices of L-shape in anti-clock-wise order, shape=(6,2)
        self.L_shape_vertices = tr.tensor([[0., 0.], [1., 0.], \
                                           [1, 0.5], [0.5, 0.5], \
                                           [0.5, 1], [0., 1.0]], \
                                          dtype=self.dtype, device=self.device)

    def A(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim,dim)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        A = tr.zeros(*x_shape_, \
                     self.dim, self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

        # define the three subdomains
        idx_sub_omega    = [None]*3
        idx_sub_omega[0] =  (x[...,0] > 0.5)*(x[...,1] > 0)\
                           *(x[...,0] < 1)  *(x[...,1] < 0.5)

        idx_sub_omega[1] =  (x[...,0] > 0)  *(x[...,1] > 0.5)\
                           *(x[...,0] < 0.5)*(x[...,1] < 1)

        idx_sub_omega[2] =  (x[...,0] > 0)  *(x[...,1] > 0)\
                           *(x[...,0] < 0.5)*(x[...,1] < 0.5)
        # subdomain 1
        A[..., 0, 0][idx_sub_omega[0]] = 2
        A[..., 1, 1][idx_sub_omega[0]] = 2
        A[..., 0, 1][idx_sub_omega[0]] = 1
        A[..., 1, 0][idx_sub_omega[0]] = 1
        # subdomain 2
        A[..., 0, 0][idx_sub_omega[1]] = 1/2
        A[..., 1, 1][idx_sub_omega[1]] = 1/2
        A[..., 0, 1][idx_sub_omega[1]] = 1/4
        A[..., 1, 0][idx_sub_omega[1]] = 1/4
        # subdomain 3
        A[..., 0, 0][idx_sub_omega[2]] = 1
        A[..., 1, 1][idx_sub_omega[2]] = 1
        A[..., 0, 1][idx_sub_omega[2]] = 1/2
        A[..., 1, 0][idx_sub_omega[2]] = 1/2

        return A 

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


    def b(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # parameter
        mu1 = 1
        mu2 = 1

        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        b = tr.zeros(*x_shape_, \
                     self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

        # define the three subdomains
        idx_sub_omega    = [None]*3
        idx_sub_omega[0] =  (x[...,0] > 0.5)*(x[...,1] > 0)\
                           *(x[...,0] < 1)  *(x[...,1] < 0.5)

        idx_sub_omega[1] =  (x[...,0] > 0)  *(x[...,1] > 0.5)\
                           *(x[...,0] < 0.5)*(x[...,1] < 1)

        idx_sub_omega[2] =  (x[...,0] > 0)  *(x[...,1] > 0)\
                           *(x[...,0] < 0.5)*(x[...,1] < 0.5)
        # subdomain 1
        b[...,0][idx_sub_omega[0]] = \
                                 mu1*(8*(x[...,1][idx_sub_omega[0]]-1/4)**2-1/2)
        # subdomain 2
        b[...,1][idx_sub_omega[1]] = \
                                 mu2*(8*(x[...,0][idx_sub_omega[1]]-1/4)**2-1/2)
        # subdomain 3
        b[...,0][idx_sub_omega[2]] = \
                                 mu1*(8*(x[...,1][idx_sub_omega[2]]-1/4)**2-1/2)
        b[...,1][idx_sub_omega[2]] = \
                                 mu2*(8*(x[...,0][idx_sub_omega[2]]-1/4)**2-1/2)

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

        # calc the function 
        result[..., 0] = 10*tr.abs(tr.sin(2*pi*x[...,0])*tr.sin(2*pi*x[...,1]))

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


    def u_sol(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # from torch to numpy
        x_np = np.zeros((3, x.shape[0]))
        x_np[:2,:] = x.detach().cpu().transpose(0,1)

        # shape (dim,...)
        u_tr = tr.tensor(self.u_sol_fenicsx(x_np), \
                         dtype=self.dtype, \
                         device=self.device).transpose(0,1)

        return u_tr


    def char_func_omega(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        
        # get the indices on L-shape
        idx_square     =    (x[...,0] > 0. )*(x[...,1] > 0.5)\
                           *(x[...,0] < 0.5)*(x[...,1] < 1.)
        idx_rectangle  = (x[...,0] > 0.)*(x[...,1] > 0)\
                        *(x[...,0] < 1.)*(x[...,1] < 0.5)
        
        # evaluate the characteristic function
        value_tr                    = tr.zeros(*list(x.shape[:-1]), \
                                               1, \
                                               dtype=self.dtype,\
                                               device=self.device)
        value_tr[...,0][idx_square]      = 1.
        value_tr[...,0][idx_rectangle]   = 1.
        
        return value_tr


    def char_func_boxWOomega(self, x):
        # returns 1 for x \in \square \setminus \Omega=L-shape and zero else
        
        # get the indices on L-shape
        idx_square     =    (x[...,0] > 0. )*(x[...,1] > 0.5)\
                           *(x[...,0] < 0.5)*(x[...,1] < 1.)
        idx_rectangle  = (x[...,0] > 0.)*(x[...,1] > 0)\
                        *(x[...,0] < 1.)*(x[...,1] < 0.5)

        idx_set = tr.logical_not(tr.logical_or(idx_square, idx_rectangle))

        # evaluate the characteristic function
        value_tr                 = tr.zeros(*list(x.shape[:-1]), \
                                            1, \
                                            dtype=self.dtype, \
                                            device=self.device)
        value_tr[...,0][idx_set] = 1.

        return value_tr


    ### Fenicsx methods
    def A_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim*dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        A_tr = self.A(x_tr)

        # reshape and torch into numpy
        A_np = tr.transpose(tr.flatten(A_tr, start_dim = 1), 0, 1).detach()\
                                                                  .numpy()

        return A_np


    def b_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        b_tr = self.b(x_tr)

        # reshape and torch into numpy
        b_np = tr.transpose(b_tr, 0, 1).detach().numpy()

        return b_np


    def c_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        c_tr = self.c(x_tr)

        # reshape and torch into numpy
        c_np = tr.transpose(c_tr, 0, 1).detach().numpy()

        return c_np


    def f_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        f_tr = self.f(x_tr)
        f_np = tr.transpose(f_tr,0,1).detach().numpy()

        return f_np


    def mean_value_potential_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        mvp_tr = self.mean_value_potential(x_tr)
        mvp_np = tr.transpose(mvp_tr,0,1).detach().numpy()

        return mvp_np


    def uD_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        uD_tr = self.uD(x_tr)

        # reshape and torch into numpy
        uD_np = tr.transpose(uD_tr, 0, 1).detach().numpy()

        return uD_np


    def u_sol_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)

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
        for i, point in enumerate(x_np):
            if len(colliding_cells.links(i)) > 0:
                actual_cells.append(colliding_cells.links(i)[0])
            else:
                ValueError("Point "+repr(i)+" is not inside the domain")

        u_sol_np = np.transpose(self.fem_sol_fnx.eval(x_np, actual_cells))

        return u_sol_np


    def char_func_omega_fenicsx(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        x_tr     = tr.transpose(tr.tensor(x[0:2,:], dtype=tr.double), 0, 1)
        
        # evaluate the char function
        value_np = tr.transpose(self.char_func_omega(x_tr),0,1).detach().numpy()
        
        return value_np


    def char_func_boxWOomega_fenicsx(self, x):
        # returns 1 for x \in \Omega=L-shape and zero else
        x_tr     = tr.transpose(tr.tensor(x[0:2,:], dtype=tr.double), 0, 1)
        
        # evaluate the char function
        value_np = tr.transpose(self.char_func_boxWOomega(x_tr),0,1)\
                                                               .detach().numpy()
        
        return value_np


    def coercivity_const(self):
        # Output: Coercivity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the coercivity constant (minimal eigenvalue of diff. matrix)

        return 1./4


    def mean_value_potential(self, x):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        # Implements the approximate distance function to impose dirichlet 
        # bdry condtions exactly. Reference:
        # [Exact imposition of boundary conditions with distance functions in
        #  physics-informed deep neural networks]

        # x shape
        x_shape_ = list(x.shape[:-1])

        # in case the pde is parametric
        x = x[...,:self.dim].reshape(-1,self.dim)

        # mean value potentials (eq. 14)
        Phi = tr.zeros(x.shape[0], \
                       1, \
                       dtype=self.dtype, \
                       device=self.device)

        # eval only points within the domain, zero otherwise
        idx_stability = tr.logical_not(\
                        tr.logical_or( (x[...,1] > 0)*(x[...,1] < 1)\
                                      *(x[...,0] > 0)*(x[...,0] < 0.5),\
                                       (x[...,1] > 0)*(x[...,1] < 0.5)\
                                      *(x[...,0] >= 0.5)*(x[...,0] < 1)))
 
        # calculate first all necessary quantities
        # data structures
        norm_r_i    = [None]*self.L_shape_vertices.shape[0]
        r_i         = [None]*self.L_shape_vertices.shape[0]
        dot_prod_r  = [None]*self.L_shape_vertices.shape[0]

        # vector from x to vertex, shape=(...,dim) 
        n_vertices = self.L_shape_vertices.shape[0]
        r_i[0]        = self.L_shape_vertices[0:1,:] - x
        norm_r_i[0]   = tr.linalg.norm(r_i[0], dim=1)

        # check if x is on an edge
        idx_on_edge = tr.isclose(norm_r_i[0], \
                         tr.zeros(1, dtype=self.dtype, device=self.device),\
                         rtol=1e-12, atol=1e-12)
        idx_stability = tr.logical_or(idx_stability, idx_on_edge)

        # -1 as tensor
        min_1 = -tr.ones(1, dtype=self.dtype, device=self.device)
        for idx, x_i in enumerate(self.L_shape_vertices[1:,:], 1):
            r_i[idx]        = x_i - x
            norm_r_i[idx]   = tr.linalg.norm(r_i[idx], dim=1)
            dot_prod_r[idx-1] = tr.matmul(r_i[idx-1].unsqueeze(1), \
                                          r_i[idx].unsqueeze(2))\
                                                        .squeeze(-1).squeeze(-1)

            # check at the same time the stability indeces
            # check if x is on an edge
            idx_on_edge = tr.isclose(norm_r_i[idx], \
                             tr.zeros(1, dtype=self.dtype, device=self.device),\
                             rtol=1e-12, atol=1e-12)
            idx_stability = tr.logical_or(idx_stability, idx_on_edge)

            # set zero to those values on the bdry
            idx_on_bdry= tr.isclose(dot_prod_r[idx-1].squeeze(-1).squeeze(-1)/\
                                 (norm_r_i[idx-1]*norm_r_i[idx]), \
                                         min_1, rtol=1e-12, atol=1e-12)
            idx_stability = tr.logical_or(idx_stability, idx_on_bdry)


        # last dot product
        dot_prod_r[-1] = tr.matmul(r_i[-1].unsqueeze(1), \
                                   r_i[0].unsqueeze(2))\
                                                    .squeeze(-1).squeeze(-1)

        # set zero to those values on the bdry
        idx_on_bdry= tr.isclose(dot_prod_r[-1].squeeze(-1).squeeze(-1)/\
                             (norm_r_i[-1]*norm_r_i[0]), \
                                     min_1, rtol=1e-12, atol=1e-12)
        idx_stability = tr.logical_or(idx_stability, idx_on_bdry)

        # first value of W
        W = 0.
        eval_idx = idx_stability == False
        for idx in range(n_vertices):
            r_i_vec   = r_i[idx][eval_idx, :]
            r_ip1_vec = r_i[(idx+1)%n_vertices][eval_idx,:]
            norm_r_i_vec    = norm_r_i[idx][eval_idx]
            norm_r_ip1_vec  = norm_r_i[(idx+1)%n_vertices][eval_idx]
            dot_prod_r_vec  = dot_prod_r[idx][eval_idx]

            # see (eq. 13), shape=(...,1)
            t_i   = tr.linalg.det(tr.cat((r_i_vec.unsqueeze(-1), \
                                          r_ip1_vec.unsqueeze(-1)), dim=2))/\
                    (  norm_r_i_vec*norm_r_ip1_vec + dot_prod_r_vec)

            # see (eq. 13), shape=(...,1)
            W     = W + (1./norm_r_i_vec + 1./norm_r_ip1_vec)*t_i


        # mean value potential
        Phi[eval_idx,0] = 2./W

        # reshape to the shape of the input x
        Phi = Phi.reshape(*x_shape_, 1)
        
        return Phi



# general elliptic test cases
class pde_general_elliptic_Lshape_Navier():
    def __init__(self, \
                 mesh_fenicsx, \
                 velocity_fenicsx, \
                 dtype=tr.double, \
                 device=tr.device('cpu')):
        # dtype and device agnostic code
        self.device = device
        self.dtype  = dtype

        # dimension 
        self.dim    = 2

        # parametric
        self.parametric = False

        # assign the fenicsx stuff
        self.msh_fnx        = mesh_fenicsx
        self.vel_fnx        = velocity_fenicsx

        # vertices of L-shape in anti-clock-wise order, shape=(6,2)
        self.L_shape_vertices = tr.tensor([[0., 0.], [1., 0.], \
                                           [1, 0.5], [0.5, 0.5], \
                                           [0.5, 1], [0., 1.0]], \
                                          dtype=self.dtype, device=self.device)


    def A(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim,dim)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        A = tr.zeros(*x_shape_, \
                     self.dim, self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

        # define the three subdomains
        idx_sub_omega    = [None]*3
        idx_sub_omega[0] =  (x[...,0] > 0.5)*(x[...,1] > 0)\
                           *(x[...,0] < 1)  *(x[...,1] < 0.5)

        idx_sub_omega[1] =  (x[...,0] > 0)  *(x[...,1] > 0.5)\
                           *(x[...,0] < 0.5)*(x[...,1] < 1)

        idx_sub_omega[2] =  (x[...,0] > 0)  *(x[...,1] > 0)\
                           *(x[...,0] < 0.5)*(x[...,1] < 0.5)
        # subdomain 1
        A[idx_sub_omega[0], 0, 0] = 2
        A[idx_sub_omega[0], 1, 1] = 2
        A[idx_sub_omega[0], 0, 1] = 1
        A[idx_sub_omega[0], 1, 0] = 1
        # subdomain 2
        A[idx_sub_omega[1], 0, 0] = 1/2
        A[idx_sub_omega[1], 1, 1] = 1/2
        A[idx_sub_omega[1], 0, 1] = 1/4
        A[idx_sub_omega[1], 1, 0] = 1/4
        # subdomain 3
        A[idx_sub_omega[2], 0, 0] = 1
        A[idx_sub_omega[2], 1, 1] = 1
        A[idx_sub_omega[2], 0, 1] = 1/2
        A[idx_sub_omega[2], 1, 0] = 1/2

        return A 

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


    def b(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # from torch to numpy
        x_np = np.zeros((3, x.shape[0]))
        x_np[:2,:] = x.transpose(0,1)

        # shape (dim,...)
        b_tr = tr.tensor(self.b_fenicsx(x_np), \
                         dtype=self.dtype, \
                         device=self.device).transpose(0,1)

        return b_tr


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

        # calc the function 
        result[..., 0] = 10*tr.abs(tr.sin(2*pi*x[...,0])*tr.sin(2*pi*x[...,1]))

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


    def char_func_omega(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        
        # get the indices on L-shape
        idx_square     =    (x[...,0] > 0. )*(x[...,1] > 0.5)\
                           *(x[...,0] < 0.5)*(x[...,1] < 1.)
        idx_rectangle  = (x[...,0] > 0.)*(x[...,1] > 0)\
                        *(x[...,0] < 1.)*(x[...,1] < 0.5)
        
        # evaluate the characteristic function
        value_tr                    = tr.zeros(x.shape[0], 1, dtype=tr.double)
        value_tr[idx_square,:]      = 1.
        value_tr[idx_rectangle,:]   = 1.
        
        return value_tr


    ### Fenicsx methods
    def A_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim*dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        A_tr = self.A(x_tr)

        # reshape and torch into numpy
        A_np = tr.transpose(tr.flatten(A_tr, start_dim = 1), 0, 1).detach()\
                                                                  .numpy()

        return A_np


    def b_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)

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
        for i, point in enumerate(x_np):
            if len(colliding_cells.links(i)) > 0:
                actual_cells.append(colliding_cells.links(i)[0])

        b_np  = np.transpose(self.vel_fnx.eval(x_np, actual_cells))

        return b_np


    def c_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        c_tr = self.c(x_tr)

        # reshape and torch into numpy
        c_np = tr.transpose(c_tr, 0, 1).detach().numpy()

        return c_np


    def f_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        f_tr = self.f(x_tr)
        f_np = tr.transpose(f_tr,0,1).detach().numpy()

        return f_np


    def uD_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        uD_tr = self.uD(x_tr)

        # reshape and torch into numpy
        uD_np = tr.transpose(uD_tr, 0, 1).detach().numpy()

        return uD_np


    def char_func_omega_fenicsx(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        x_tr     = tr.transpose(tr.tensor(x[0:2,:], dtype=tr.double), 0, 1)
        
        # evaluate the char function
        value_np = tr.transpose(self.char_func_omega(x_tr),0,1).detach().numpy()
        
        return value_np


    def coercivity_const(self):
        # Output: Coercivity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the coercivity constant (minimal eigenvalue of diff. matrix)

        return 1./4


    def mean_value_potential(self, x):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        # Implements the approximate distance function to impose dirichlet 
        # bdry condtions exactly. Reference:
        # [Exact imposition of boundary conditions with distance functions in
        #  physics-informed deep neural networks]

        # x shape
        x_shape_ = list(x.shape[:-1])

        # in case the pde is parametric
        x = x[...,:self.dim].reshape(-1,self.dim)

        # mean value potentials (eq. 14)
        Phi = tr.zeros(x.shape[0], \
                       1, \
                       dtype=self.dtype, \
                       device=self.device)

        # eval only points within the domain, zero otherwise
        idx_stability = tr.logical_not(\
                        tr.logical_or( (x[...,1] > 0)*(x[...,1] < 1)\
                                      *(x[...,0] > 0)*(x[...,0] < 0.5),\
                                       (x[...,1] > 0)*(x[...,1] < 0.5)\
                                      *(x[...,0] >= 0.5)*(x[...,0] < 1)))
 
        # calculate first all necessary quantities
        # data structures
        norm_r_i    = [None]*self.L_shape_vertices.shape[0]
        r_i         = [None]*self.L_shape_vertices.shape[0]
        dot_prod_r  = [None]*self.L_shape_vertices.shape[0]

        # vector from x to vertex, shape=(...,dim) 
        n_vertices = self.L_shape_vertices.shape[0]
        r_i[0]        = self.L_shape_vertices[0:1,:] - x
        norm_r_i[0]   = tr.linalg.norm(r_i[0], dim=1)

        # check if x is on an edge
        idx_on_edge = tr.isclose(norm_r_i[0], \
                         tr.zeros(1, dtype=self.dtype, device=self.device),\
                         rtol=1e-12, atol=1e-12)
        idx_stability = tr.logical_or(idx_stability, idx_on_edge)

        # -1 as tensor
        min_1 = -tr.ones(1, dtype=self.dtype, device=self.device)
        for idx, x_i in enumerate(self.L_shape_vertices[1:,:], 1):
            r_i[idx]        = x_i - x
            norm_r_i[idx]   = tr.linalg.norm(r_i[idx], dim=1)
            dot_prod_r[idx-1] = tr.matmul(r_i[idx-1].unsqueeze(1), \
                                          r_i[idx].unsqueeze(2))\
                                                        .squeeze(-1).squeeze(-1)

            # check at the same time the stability indeces
            # check if x is on an edge
            idx_on_edge = tr.isclose(norm_r_i[idx], \
                             tr.zeros(1, dtype=self.dtype, device=self.device),\
                             rtol=1e-12, atol=1e-12)
            idx_stability = tr.logical_or(idx_stability, idx_on_edge)

            # set zero to those values on the bdry
            idx_on_bdry= tr.isclose(dot_prod_r[idx-1].squeeze(-1).squeeze(-1)/\
                                 (norm_r_i[idx-1]*norm_r_i[idx]), \
                                         min_1, rtol=1e-12, atol=1e-12)
            idx_stability = tr.logical_or(idx_stability, idx_on_bdry)


        # last dot product
        dot_prod_r[-1] = tr.matmul(r_i[-1].unsqueeze(1), \
                                   r_i[0].unsqueeze(2))\
                                                    .squeeze(-1).squeeze(-1)

        # set zero to those values on the bdry
        idx_on_bdry= tr.isclose(dot_prod_r[-1].squeeze(-1).squeeze(-1)/\
                             (norm_r_i[-1]*norm_r_i[0]), \
                                     min_1, rtol=1e-12, atol=1e-12)
        idx_stability = tr.logical_or(idx_stability, idx_on_bdry)

        # first value of W
        W = 0.
        eval_idx = idx_stability == False
        for idx in range(n_vertices):
            r_i_vec   = r_i[idx][eval_idx, :]
            r_ip1_vec = r_i[(idx+1)%n_vertices][eval_idx,:]
            norm_r_i_vec    = norm_r_i[idx][eval_idx]
            norm_r_ip1_vec  = norm_r_i[(idx+1)%n_vertices][eval_idx]
            dot_prod_r_vec  = dot_prod_r[idx][eval_idx]

            # see (eq. 13), shape=(...,1)
            t_i   = tr.linalg.det(tr.cat((r_i_vec.unsqueeze(-1), \
                                          r_ip1_vec.unsqueeze(-1)), dim=2))/\
                    (  norm_r_i_vec*norm_r_ip1_vec + dot_prod_r_vec)

            # see (eq. 13), shape=(...,1)
            W     = W + (1./norm_r_i_vec + 1./norm_r_ip1_vec)*t_i


        # mean value potential
        Phi[eval_idx,0] = 2./W

        # reshape to the shape of the input x
        Phi = Phi.reshape(*x_shape_, 1)
        
        return Phi


# general elliptic test cases
class pde_general_elliptic_sawblade():
    def __init__(self, \
                 n_sawtooth, \
                 ampli,\
                 y_offset, \
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
        self.dim    = 2

        # parametric
        self.parametric = True
        self.P_space    = parameter_space

        # assign the fenicsx stuff
        self.msh_fnx       = mesh_fenicsx
        self.fem_sol_fnx   = fem_sol_fenicsx
        # shape=(p, fem dimension)
        self.fem_sol_data  = fem_sol_data

        # assign some parameters which descripes the domain
        self.start_x     = 0
        self.end_x       = 4
        self.n_sawtooth  = n_sawtooth 
        self.ncoords     = 2*n_sawtooth + 1
        self.ampli       = ampli
        self.loc         = tr.linspace(self.start_x, self.end_x, self.ncoords, \
                                       dtype=self.dtype, device=self.device)
        self.h           = self.loc[1] - self.loc[0]
        self.y_offset    = y_offset 
        self.slope       = (self.ampli-self.y_offset)/self.h

        # vertices of sawblade in anti-clock-wise order,shape=(3+2*n_sawthooth,2)
        # create the vertices of the saw tooths
        saw_tooth_vert = tr.cat((self.loc[:-1].reshape(-1,1), \
                                      tr.ones(self.ncoords-1,1,\
                                              dtype=self.dtype,\
                                              device=self.device)), dim=1)
        saw_tooth_vert[0:-1:2,1] = 0.5
        self.saw_blade_vertices = tr.tensor([[0., 0], [4., 0.], [4., 0.5]], \
                                          dtype=self.dtype, device=self.device)
        self.saw_blade_vertices = tr.cat((self.saw_blade_vertices, \
                                        tr.flipud(saw_tooth_vert)), dim=0)


    def A(self, x, mu=[0.1, 3], **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim,dim)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        A_diff = tr.zeros(*x_shape_, \
                     self.dim, self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

        # calc the idx of the tooth
        y_tooth     = self.sawblade(x[...,0:1])
        idx_tooth   = (x[...,1]>self.y_offset)*(x[...,1] < y_tooth[...,0])\
                      *(x[...,0] > self.start_x)*(x[...,0] < self.end_x)

        # define the three subdomains
        # on the tooth
        A_diff[..., 0, 0][idx_tooth] =   mu[0]
        A_diff[..., 1, 1][idx_tooth] = 2*mu[0]

        # below tooth
        # calc the idx of below the tooth
        idx_blade = (x[...,1]<self.y_offset)*(x[...,1]>0.)\
                   *(x[...,0] > self.start_x)*(x[...,0] < self.end_x)
        A_diff[..., 0, 0][idx_blade] =   mu[1]
        A_diff[..., 1, 1][idx_blade] = 2*mu[1]

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


    def b(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        b = tr.zeros(*x_shape_, \
                     self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

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

        return result


    def f(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        
        # index structure
        #y_tooth     = self.sawblade(x[...,0:1])
        #idx_blade   =    (x[...,1]>0)*(x[...,1] < y_tooth[...,0])\
        #                *(x[...,0]>self.start_x)*(x[...,0] < self.end_x)

        # evaluate the characteristic function
        # x shape
        x_shape_ = list(x.shape[:-1])
        # data structure
        result = tr.ones(*x_shape_, \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        #result[...,0][idx_blade] = 1.
 
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
        x_np[:2,:] = x.detach().cpu().transpose(0,1)

        # shape (dim,...)
        u_tr = tr.tensor(self.u_sol_fenicsx(x_np, mu_idx=mu_idx), \
                         dtype=self.dtype, \
                         device=self.device).transpose(0,1)

        return u_tr


    def char_func_omega(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        
        # index structure
        y_tooth     = self.sawblade(x[...,0:1])
        idx_blade   =    (x[...,1]>0)*(x[...,1] < y_tooth[...,0])\
                        *(x[...,0]>self.start_x)*(x[...,0] < self.end_x)

        # evaluate the characteristic function
        # x shape
        x_shape_ = list(x.shape[:-1])
        # data structure
        result = tr.zeros(*x_shape_, \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        result[...,0][idx_blade] = 1.
        
        return result


    def char_func_boxWOomega(self, x):
        # returns 1 for x \in \Omega=L-shape and zero else
        
        # index structure
        y_tooth     = self.sawblade(x[...,0:1])
        idx_blade   =    (x[...,1]>0)*(x[...,1] < y_tooth[...,0])\
                        *(x[...,0]>self.start_x)*(x[...,0] < self.end_x)
       
        # evaluate the characteristic function
        # x shape
        x_shape_ = list(x.shape[:-1])
        # data structure
        result = tr.zeros(*x_shape_, \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        result[...,0][tr.logical_not(idx_blade)] = 1.
        
        return result


    ### Fenicsx methods
    def A_fenicsx(self, x, mu=[0.1, 3], **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim*dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        A_tr = self.A(x_tr, mu)

        # reshape and torch into numpy
        A_np = tr.transpose(tr.flatten(A_tr, start_dim = 1), 0, 1).detach()\
                                                                  .numpy()

        return A_np


    def b_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        b_tr = self.b(x_tr)

        # reshape and torch into numpy
        b_np = tr.transpose(b_tr, 0, 1).detach().numpy()

        return b_np


    def c_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        c_tr = self.c(x_tr)

        # reshape and torch into numpy
        c_np = tr.transpose(c_tr, 0, 1).detach().numpy()

        return c_np


    def f_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        f_tr = self.f(x_tr)
        f_np = tr.transpose(f_tr,0,1).detach().numpy()

        return f_np


    def uD_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

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
        for i, point in enumerate(x_np):
            if len(colliding_cells.links(i)) > 0:
                actual_cells.append(colliding_cells.links(i)[0])
            else:
                raise ValueError("Point "+repr(i)+" is not inside the domain")
        u_sol_np = np.transpose(self.fem_sol_fnx.eval(x_np, actual_cells))

        return u_sol_np


    def mean_value_potential_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        mvp_tr = self.mean_value_potential(x_tr)
        mvp_np = tr.transpose(mvp_tr,0,1).detach().numpy()

        return mvp_np


    def char_func_omega_fenicsx(self, x):
        # returns 1 for x \in \Omega=L-shape and zero else
        x_tr     = tr.transpose(tr.tensor(x[0:2,:], dtype=tr.double), 0, 1)
        
        # evaluate the char function
        value_np = tr.transpose(self.char_func_omega(x_tr),0,1).detach().numpy()
        
        return value_np


    def char_func_boxWOomega_fenicsx(self, x):
        # returns 1 for x \in \Omega=L-shape and zero else
        x_tr     = tr.transpose(tr.tensor(x[0:2,:], dtype=tr.double), 0, 1)
        
        # evaluate the char function
        value_np = tr.transpose(self.char_func_boxWOomega(x_tr),0,1)\
                                                               .detach().numpy()
        
        return value_np


    def coercivity_const(self, mu):
        # Output: Coercivity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the coercivity constant (minimal eigenvalue of diff. matrix)

        return tr.minimum(mu[0], mu[1])


    def continuity_const(self, mu):
        # Output: Continuity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the continuity constant (maximal eigenvalue of diff. matrix)

        return 2*tr.maximum(mu[0], mu[1])


    def sawblade(self, x):
        # Input: x.shape = (..., 1), (pytorch tensor)
        # Output: out.shape = (..., 1), (pytorch tensor)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        result = (0.5-1e-10)*tr.ones(*x_shape_, \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        ones_temp = tr.ones(1, dtype=self.dtype, device=self.device)

        for i in range(self.n_sawtooth):
            idx_tooth = (x > self.loc[2*i])*(x < self.loc[2*i+2])
            result[idx_tooth] = tr.maximum(self.ampli-self.slope*\
                                          tr.abs(x[idx_tooth]-self.loc[2*i+1]),\
                                          self.y_offset*ones_temp)

        return result


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
        zeros_tensor = tr.zeros(1, dtype=self.dtype, device=self.device)

        # x shape
        x_shape_ = list(x.shape[:-1])

        # mean value potentials (eq. 14), shape=(x.shape[:-1], 1)
        Phi = tr.zeros(x[...,:2].reshape(-1,2).shape[0], \
                       1, \
                       dtype=self.dtype, \
                       device=self.device)

        # difference vectors for all vertices, shape=(N_vert, N_coords, 2)
        R_i =     self.saw_blade_vertices.unsqueeze(1) \
                - x[...,:2].reshape(-1,2).unsqueeze(0)

        # get the norm of the difference vectors, shape=(N_vert, N_coords)
        norm_R_i = tr.linalg.norm(R_i, dim=2)
        idx_stab = tr.isclose(norm_R_i, zeros_tensor, rtol=rtol, atol=atol)

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
        idx_stab       = tr.logical_or(idx_stab, 
                                      tr.isclose(det_R_i, 
                                                 zeros_tensor, 
                                                 rtol=rtol, atol=atol))
        # get the product of the norm of r_i and norm of r_i+1
        # , shape=(N_vert, N_spatial)
        prod_R_i = tr.zeros(norm_R_i.shape, 
                            dtype=self.dtype, 
                            device=self.device)
        prod_R_i[:-1,:] = norm_R_i[:-1,:]*norm_R_i[1:,:]
        prod_R_i[-1,:]  = norm_R_i[-1,:]*norm_R_i[0,:]

        # get t_i step by step, shape=(N_vert, N_coords)
        t_i          = prod_R_i + scal_prod_R_i
        idx_stab      = tr.logical_or(idx_stab, 
                                      tr.isclose(t_i, 
                                                 zeros_tensor, 
                                                 rtol=rtol, atol=atol))
        t_i           = 1./t_i
        t_i           = det_R_i*t_i
        idx_stab      = tr.logical_or(idx_stab, 
                                      tr.isclose(t_i, 
                                                 zeros_tensor, 
                                                 rtol=rtol, atol=atol))

        # get 1/norm(R_i)
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

        # reshape to the shape of the input x
        Phi = Phi.reshape(*x_shape_, 1)

        return Phi




# general elliptic test cases
class pde_general_elliptic_thermalblock():
    def __init__(self, \
                 n_sawtooth, \
                 ampli,\
                 y_offset, \
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
        self.dim    = 2

        # parametric
        self.parametric = True
        self.P_space    = parameter_space

        # assign the fenicsx stuff
        self.msh_fnx       = mesh_fenicsx
        self.fem_sol_fnx   = fem_sol_fenicsx
        # shape=(p, fem dimension)
        self.fem_sol_data  = fem_sol_data

        # assign some parameters which descripes the domain
        self.start_x     = 0
        self.end_x       = 4
        self.n_sawtooth  = n_sawtooth 
        self.ncoords     = 2*n_sawtooth + 1
        self.ampli       = ampli
        self.loc         = tr.linspace(self.start_x, self.end_x, self.ncoords, \
                                       dtype=self.dtype, device=self.device)
        self.h           = self.loc[1] - self.loc[0]
        self.y_offset    = y_offset 
        self.slope       = (self.ampli-self.y_offset)/self.h

        # vertices of sawblade in anti-clock-wise order,shape=(3+2*n_sawthooth,2)
        # create the vertices of the saw tooths
        saw_tooth_vert = tr.cat((self.loc[:-1].reshape(-1,1), \
                                      tr.ones(self.ncoords-1,1,\
                                              dtype=self.dtype,\
                                              device=self.device)), dim=1)
        saw_tooth_vert[0:-1:2,1] = 0.5
        self.saw_blade_vertices = tr.tensor([[0., 0], [4., 0.], [4., 0.5]], \
                                          dtype=self.dtype, device=self.device)
        self.saw_blade_vertices = tr.cat((self.saw_blade_vertices, \
                                        tr.flipud(saw_tooth_vert)), dim=0)


    def A(self, x, mu=[1., 1., 1., 1.], **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim,dim)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        A_diff = tr.zeros(*x_shape_, \
                     self.dim, self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

        # calc the idx of the tooth
        y_tooth          = self.sawblade(x[...,0:1])
        idx_tooth_block3 = (x[...,1]>self.y_offset)*(x[...,1] < y_tooth[...,0])\
                          *(x[...,0] > self.start_x)*(x[...,0] < self.end_x/2)

        # define the three subdomains
        # on the tooth
        A_diff[..., 0, 0][idx_tooth_block3] =   mu[2]
        A_diff[..., 1, 1][idx_tooth_block3] = 2*mu[2]

        idx_tooth_block4 = (x[...,1]>self.y_offset)*(x[...,1] < y_tooth[...,0])\
                          *(x[...,0] > self.end_x/2)*(x[...,0] < self.end_x)

        # define the three subdomains
        # on the tooth
        A_diff[..., 0, 0][idx_tooth_block4] =   mu[3]
        A_diff[..., 1, 1][idx_tooth_block4] = 2*mu[3]

        # below tooth
        # calc the idx of below the tooth
        idx_blade_block1 = (x[...,1]<self.y_offset)*(x[...,1]>0.)\
                   *(x[...,0] > self.start_x)*(x[...,0] < self.end_x/2)
        A_diff[..., 0, 0][idx_blade_block1] =   mu[0]
        A_diff[..., 1, 1][idx_blade_block1] = 2*mu[0]

        # below tooth
        # calc the idx of below the tooth
        idx_blade_block2 = (x[...,1]<self.y_offset)*(x[...,1]>0.)\
                   *(x[...,0] > self.end_x/2)*(x[...,0] < self.end_x)
        A_diff[..., 0, 0][idx_blade_block2] =   mu[1]
        A_diff[..., 1, 1][idx_blade_block2] = 2*mu[1]

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


    def b(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        b = tr.zeros(*x_shape_, \
                     self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

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

        return result


    def f(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        
        # index structure
        #y_tooth     = self.sawblade(x[...,0:1])
        #idx_blade   =    (x[...,1]>0)*(x[...,1] < y_tooth[...,0])\
        #                *(x[...,0]>self.start_x)*(x[...,0] < self.end_x)

        # evaluate the characteristic function
        # x shape
        x_shape_ = list(x.shape[:-1])
        # data structure
        result = tr.ones(*x_shape_, \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        #result[...,0][idx_blade] = 1.
 
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
        x_np[:2,:] = x.detach().cpu().transpose(0,1)

        # shape (dim,...)
        u_tr = tr.tensor(self.u_sol_fenicsx(x_np, mu_idx=mu_idx), \
                         dtype=self.dtype, \
                         device=self.device).transpose(0,1)

        return u_tr


    def char_func_omega(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        
        # index structure
        y_tooth     = self.sawblade(x[...,0:1])
        idx_blade   =    (x[...,1]>0)*(x[...,1] < y_tooth[...,0])\
                        *(x[...,0]>self.start_x)*(x[...,0] < self.end_x)

        # evaluate the characteristic function
        # x shape
        x_shape_ = list(x.shape[:-1])
        # data structure
        result = tr.zeros(*x_shape_, \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        result[...,0][idx_blade] = 1.
        
        return result


    ### Fenicsx methods
    def A_fenicsx(self, x, mu=[0.1, 3], **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim*dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        A_tr = self.A(x_tr, mu)

        # reshape and torch into numpy
        A_np = tr.transpose(tr.flatten(A_tr, start_dim = 1), 0, 1).detach()\
                                                                  .numpy()

        return A_np


    def b_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        b_tr = self.b(x_tr)

        # reshape and torch into numpy
        b_np = tr.transpose(b_tr, 0, 1).detach().numpy()

        return b_np


    def c_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        c_tr = self.c(x_tr)

        # reshape and torch into numpy
        c_np = tr.transpose(c_tr, 0, 1).detach().numpy()

        return c_np


    def f_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        f_tr = self.f(x_tr)
        f_np = tr.transpose(f_tr,0,1).detach().numpy()

        return f_np


    def uD_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

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
        for i, point in enumerate(x_np):
            if len(colliding_cells.links(i)) > 0:
                actual_cells.append(colliding_cells.links(i)[0])
            else:
                raise ValueError("Point "+repr(i)+" is not inside the domain")
        u_sol_np = np.transpose(self.fem_sol_fnx.eval(x_np, actual_cells))

        return u_sol_np


    def mean_value_potential_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        mvp_tr = self.mean_value_potential(x_tr)
        mvp_np = tr.transpose(mvp_tr,0,1).detach().numpy()

        return mvp_np


    def char_func_omega_fenicsx(self, x):
        # returns 1 for x \in \Omega=L-shape and zero else
        x_tr     = tr.transpose(tr.tensor(x[0:2,:], dtype=tr.double), 0, 1)
        
        # evaluate the char function
        value_np = tr.transpose(self.char_func_omega(x_tr),0,1).detach().numpy()
        
        return value_np



    def coercivity_const(self, mu):
        # Output: Coercivity constant w.r.t. H10-norm, out.shape = (1,)

        # return the coercivity constant (minimal eigenvalue of diff. matrix)

        return tr.min(mu)


    def continuity_const(self, mu):
        # Output: Continuity constant w.r.t. H10-norm, out.shape = (1,)

        # return the continuity constant (maximal eigenvalue of diff. matrix)

        return 2*tr.max(mu)


    def sawblade(self, x):
        # Input: x.shape = (..., 1), (pytorch tensor)
        # Output: out.shape = (..., 1), (pytorch tensor)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        result = (0.5-1e-10)*tr.ones(*x_shape_, \
                          1, \
                          dtype=self.dtype, \
                          device=self.device)

        ones_temp = tr.ones(1, dtype=self.dtype, device=self.device)

        for i in range(self.n_sawtooth):
            idx_tooth = (x > self.loc[2*i])*(x < self.loc[2*i+2])
            result[idx_tooth] = tr.maximum(self.ampli-self.slope*\
                                          tr.abs(x[idx_tooth]-self.loc[2*i+1]),\
                                          self.y_offset*ones_temp)

        return result


    def mean_value_potential(self, x):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        # Implements the approximate distance function to impose dirichlet 
        # bdry condtions exactly. Reference:
        # [Exact imposition of boundary conditions with distance functions in
        #  physics-informed deep neural networks]

        # relative and absolute tolerance for stability
        rel_tol = 1e-12
        abs_tol = 1e-12

        # x shape
        x_shape_ = list(x.shape[:-1])

        # in case the pde is parametric
        x = x[...,:self.dim].reshape(-1,self.dim)

        # mean value potentials (eq. 14)
        Phi = tr.zeros(x.shape[0], \
                       1, \
                       dtype=self.dtype, \
                       device=self.device)

        # get the points on the bdry
        y_tooth       = self.sawblade(x[...,0:1])
        idx_stability = tr.logical_not(  \
                          (x[...,1]>0)*(x[...,1] < y_tooth[...,0])\
                         *(x[...,0]>self.start_x)*(x[...,0] < self.end_x))
 
        # calculate first all necessary quantities
        # data structures
        norm_r_i    = [None]*self.saw_blade_vertices.shape[0]
        r_i         = [None]*self.saw_blade_vertices.shape[0]
        dot_prod_r  = [None]*self.saw_blade_vertices.shape[0]

        # vector from x to vertex, shape=(...,dim) 
        n_vertices    = self.saw_blade_vertices.shape[0]
        r_i[0]        = self.saw_blade_vertices[0:1,:] - x
        norm_r_i[0]   = tr.linalg.norm(r_i[0], dim=1)
        # check if x is on an edge
        idx_on_edge = tr.isclose(norm_r_i[0], \
                            tr.zeros(1, dtype=self.dtype, device=self.device),\
                            rtol=rel_tol, atol=abs_tol)
        idx_stability = tr.logical_or(idx_stability, idx_on_edge)
        # -1 as tensor
        min_1 = -tr.ones(1, dtype=self.dtype, device=self.device)
        for idx, x_i in enumerate(self.saw_blade_vertices[1:,:], 1):
            r_i[idx]        = x_i - x
            norm_r_i[idx]   = tr.linalg.norm(r_i[idx], dim=1)
            dot_prod_r[idx-1] = tr.matmul(r_i[idx-1].unsqueeze(1), \
                                          r_i[idx].unsqueeze(2))\
                                                        .squeeze(-1).squeeze(-1)

            # check at the same time the stability indeces
            # check if x is on an edge
            idx_on_edge = tr.isclose(norm_r_i[idx], \
                             tr.zeros(1, dtype=self.dtype, device=self.device),\
                             rtol=rel_tol, atol=abs_tol)
            idx_stability = tr.logical_or(idx_stability, idx_on_edge)

            # set zero to those values on the bdry
            idx_on_bdry= tr.isclose(dot_prod_r[idx-1].squeeze(-1).squeeze(-1)/\
                                 (norm_r_i[idx-1]*norm_r_i[idx]), \
                                         min_1, rtol=rel_tol, atol=abs_tol)
            idx_stability = tr.logical_or(idx_stability, idx_on_bdry)


        # last dot product
        dot_prod_r[-1] = tr.matmul(r_i[-1].unsqueeze(1), \
                                   r_i[0].unsqueeze(2))\
                                                    .squeeze(-1).squeeze(-1)

        # set zero to those values on the bdry
        idx_on_bdry= tr.isclose(dot_prod_r[-1].squeeze(-1).squeeze(-1)/\
                             (norm_r_i[-1]*norm_r_i[0]), \
                                     min_1, rtol=rel_tol, atol=abs_tol)
        idx_stability = tr.logical_or(idx_stability, idx_on_bdry)

        # first value of W
        W = 0.
        eval_idx = idx_stability == False
        for idx in range(n_vertices):
            r_i_vec   = r_i[idx][eval_idx, :]
            r_ip1_vec = r_i[(idx+1)%n_vertices][eval_idx,:]
            norm_r_i_vec    = norm_r_i[idx][eval_idx]
            norm_r_ip1_vec  = norm_r_i[(idx+1)%n_vertices][eval_idx]
            dot_prod_r_vec  = dot_prod_r[idx][eval_idx]

            # see (eq. 13), shape=(...,1)
            t_i   = tr.linalg.det(tr.cat((r_i_vec.unsqueeze(-1), \
                                          r_ip1_vec.unsqueeze(-1)), dim=2))/\
                    (  norm_r_i_vec*norm_r_ip1_vec + dot_prod_r_vec)

            # see (eq. 13), shape=(...,1)
            W     = W + (1./norm_r_i_vec + 1./norm_r_ip1_vec)*t_i


        # mean value potential
        Phi[eval_idx,0] = 2./W

        # reshape to the shape of the input x
        Phi = Phi.reshape(*x_shape_, 1)
        
        return Phi


# general elliptic test cases
class pde_general_elliptic_parShape():
    def __init__(self, \
                 fem_sol_torch=None, \
                 fem_sol_fenicsx=None, \
                 dtype=tr.double, \
                 device=tr.device('cpu')):
        # dtype and device agnostic code
        self.device = device
        self.dtype  = dtype

        # dimension 
        self.dim    = 2

        # parametric
        self.parametric = True

        # assign the fenicsx and torch stuff
        self.fem_sol_fnx = fem_sol_fenicsx
        self.fem_sol_tr  = fem_sol_torch

        # parameter space discretization
        self.parameter_set = tr.linspace(0, \
                                         pi/2, \
                                         9, \
                                         dtype=self.dtype, \
                                         device=self.device).reshape(-1,1)

        # vertices of L-shape in anti-clock-wise order, shape=(6,2)
        self.shape_vertices = [tr.tensor([[0, 1, 1, 0],\
                                          [0, 0, 1, 1]], \
                                         dtype=self.dtype, \
                                         device=self.device).transpose(0,1),\
                          tr.tensor([[0, 0.25, 0.75, 0.75, 1, 1, 0],\
                                     [0, 0, 0.0625, 0, 0, 1, 1]], \
                                    dtype=self.dtype, \
                                    device=self.device).transpose(0,1),\
                          tr.tensor([[0, 0.25, 0.75, 0.75, 1, 1, 0],\
                                     [0, 0, 0.125, 0, 0, 1, 1]], \
                                    dtype=self.dtype, \
                                    device=self.device).transpose(0,1),\
                          tr.tensor([[0, 0.25, 0.75, 0.75, 1, 1, 0],\
                                     [0, 0, 0.1875, 0, 0, 1, 1]], \
                                    dtype=self.dtype, \
                                    device=self.device).transpose(0,1),\
                          tr.tensor([[0, 0.25, 0.75, 0.75, 1, 1, 0],\
                                     [0, 0, 0.25, 0, 0, 1, 1]], \
                                    dtype=self.dtype, \
                                    device=self.device).transpose(0,1),\
                         tr.tensor([[0, 0.25, 0.75-0.125, 0.75, 0.75, 1, 1, 0],\
                                    [0, 0, 0.25, 0.25, 0, 0, 1, 1]], \
                                   dtype=self.dtype, \
                                   device=self.device).transpose(0,1),\
                         tr.tensor([[0, 0.25, 0.75-0.25, 0.75, 0.75, 1, 1, 0],\
                                    [0, 0, 0.25, 0.25, 0, 0, 1, 1]], \
                                   dtype=self.dtype, \
                                   device=self.device).transpose(0,1),\
                         tr.tensor([[0, 0.25, 0.75-0.375, 0.75, 0.75, 1, 1, 0],\
                                    [0, 0, 0.25, 0.25, 0, 0, 1, 1]], \
                                   dtype=self.dtype, \
                                   device=self.device).transpose(0,1),\
                         tr.tensor([[0, 0.25, 0.25, 0.75, 0.75, 1, 1, 0],\
                                    [0, 0, 0.25, 0.25, 0, 0, 1, 1]], \
                                   dtype=self.dtype, \
                                   device=self.device).transpose(0,1)]


    def A(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim,dim)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        A = tr.zeros(*x_shape_, \
                     self.dim, self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

        # diffusion
        A[..., 0, 0] = 1/2
        A[..., 1, 1] = 1/2
        A[..., 0, 1] = 1/4
        A[..., 1, 0] = 1/4

        return A 

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


    def b(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        b = tr.zeros(*x_shape_, \
                     self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

        # b = (10,0)**T
        b[...,0] = 10.
        b[...,1] = -3.

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

        # calc the function 
        result[..., 0] = 10

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


    def u_sol_shapeOpt(self, x, fem_sol_fnx, msh_fnx, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # from torch to numpy
        x_np = np.zeros((3, x.shape[0]))
        x_np[:2,:] = x.detach().cpu().transpose(0,1)

        # shape (dim,...)
        u_tr = tr.tensor(self.u_sol_fenicsx(x_np, fem_sol_fnx, msh_fnx), \
                         dtype=self.dtype, \
                         device=self.device).transpose(0,1)

        return u_tr


    def line_func(self, x, mu_idx=0):
        if mu_idx == 0:
            # slope
            m =   (self.shape_vertices[mu_idx][1,1]\
                 - self.shape_vertices[mu_idx][0,1])/\
                  (self.shape_vertices[mu_idx][1,0]\
                 - self.shape_vertices[mu_idx][0,0])

            return m*x[...,0] - m*self.shape_vertices[mu_idx][0,0]
           
        else:
            # slope
            m =  (self.shape_vertices[mu_idx][2,1]\
                - self.shape_vertices[mu_idx][1,1])/\
                 (self.shape_vertices[mu_idx][2,0]\
                - self.shape_vertices[mu_idx][1,0])

            return m*x[...,0] - m*self.shape_vertices[mu_idx][1,0]


    def char_func_omega(self, x, mu, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        
        # get the mu idx
        mu_idx = tr.nonzero(tr.isclose(mu, self.parameter_set))[0,0]

        # get the indices on L-shape
        idx_square  =    (x[...,0] > 0. )*(x[...,1] > 0.)\
                        *(x[...,0] < 1.0)*(x[...,1] < 1.)

        idx_cutout  = (x[...,0] >= 0.25)\
                     *(x[...,0] <= 0.75)\
                     *(x[...,1] <= 0.25)\
                     *(x[...,1] >= 0.)

        # mu dependent triangle
        idx_line  = x[...,1] > self.line_func(x, mu_idx=mu_idx)

        # build the indices
        eval_idx = tr.logical_or(tr.logical_and(idx_square, \
                                                tr.logical_not(idx_cutout)),
                                 idx_line)

        # evaluate the characteristic function
        value_tr = tr.zeros(*list(x.shape[:-1]), \
                            1, \
                            dtype=self.dtype,\
                            device=self.device)

        value_tr[...,0][eval_idx] = 1.
        
        return value_tr


    ### Fenicsx methods
    def A_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim*dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        A_tr = self.A(x_tr)

        # reshape and torch into numpy
        A_np = tr.transpose(tr.flatten(A_tr, start_dim = 1), 0, 1).detach()\
                                                                  .numpy()

        return A_np


    def b_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        b_tr = self.b(x_tr)

        # reshape and torch into numpy
        b_np = tr.transpose(b_tr, 0, 1).detach().numpy()

        return b_np


    def c_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        c_tr = self.c(x_tr)

        # reshape and torch into numpy
        c_np = tr.transpose(c_tr, 0, 1).detach().numpy()

        return c_np


    def f_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        f_tr = self.f(x_tr)
        f_np = tr.transpose(f_tr,0,1).detach().numpy()

        return f_np


    def mean_value_potential_fenicsx(self, x, mu=0., **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        mvp_tr = self.mean_value_potential(x_tr, mu=mu)
        mvp_np = tr.transpose(mvp_tr,0,1).detach().numpy()

        return mvp_np


    def uD_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        uD_tr = self.uD(x_tr)

        # reshape and torch into numpy
        uD_np = tr.transpose(uD_tr, 0, 1).detach().numpy()

        return uD_np


    def u_sol_fenicsx(self, x, fem_sol_fnx, msh_fnx, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)

        # get x in the right shape and calc the tree
        x_np = np.transpose(x)
        tree = bb_tree(msh_fnx, msh_fnx.geometry.dim)

        # compute collisions
        cell_candidates = compute_collisions_points(tree, x_np)
        colliding_cells = compute_colliding_cells(msh_fnx, \
                                                  cell_candidates, \
                                                  x_np)
        # now we have to loop over the cells 
        actual_cells = []
        for i, point in enumerate(x_np):
            if len(colliding_cells.links(i)) > 0:
                actual_cells.append(colliding_cells.links(i)[0])
            else:
                raise ValueError("Point "+repr(i)+" is not inside the domain")

        u_sol_np = np.transpose(fem_sol_fnx.eval(x_np, actual_cells))

        return u_sol_np


    def char_func_omega_fenicsx(self, x, mu=0, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        x_tr     = tr.transpose(tr.tensor(x[0:2,:], dtype=tr.double), 0, 1)
        
        # evaluate the char function
        value_np = tr.transpose(self.char_func_omega(x_tr, mu=mu),0,1)\
                                                               .detach().numpy()
        
        return value_np


    def coercivity_const(self, mu=0.):
        # Output: Coercivity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the coercivity constant (minimal eigenvalue of diff. matrix)

        return 1./4


    def continuity_const(self, mu=0.):
        # Output: Continuity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the continuity constant 
        #   (=||A||_\infty + ||b||_\infty + ||c||_infty)

        return 1./2 + 10. + 2.


    def mean_value_potential(self, \
                             x, \
                             mu=tr.zeros(1, 1, \
                                         dtype=tr.double, \
                                         device=tr.device('cpu')), \
                             **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        # Implements the approximate distance function to impose dirichlet 
        # bdry condtions exactly. Reference:
        # [Exact imposition of boundary conditions with distance functions in
        #  physics-informed deep neural networks]

        # get the mu idx
        mu_idx = tr.nonzero(tr.isclose(mu, self.parameter_set))[0,0]

        # the shape vertices of param dependent domain
        shape_vertices = self.shape_vertices[mu_idx]

        # x shape
        x_shape_ = list(x.shape[:-1])

        # in case the pde is parametric
        x = x[...,:self.dim].reshape(-1,self.dim)

        # mean value potentials (eq. 14)
        Phi = tr.zeros(x.shape[0], \
                       1, \
                       dtype=self.dtype, \
                       device=self.device)

        # eval only points within the domain, zero otherwise
        # get the indices on L-shape
        idx_square  =    (x[...,0] > 0. )*(x[...,1] > 0.)\
                        *(x[...,0] < 1.0)*(x[...,1] < 1.)

        idx_cutout  = (x[...,0] >= 0.25)\
                     *(x[...,0] <= 0.75)\
                     *(x[...,1] <= 0.25)

        # mu dependent triangle
        idx_line  = x[...,1] > self.line_func(x, mu_idx=mu_idx) 

        # build the indices
        eval_idx = tr.logical_or(tr.logical_and(idx_square, \
                                                tr.logical_not(idx_cutout)),
                                 idx_line)


        idx_stability = tr.logical_not(eval_idx)
 
        # calculate first all necessary quantities
        # data structures
        norm_r_i    = [None]*shape_vertices.shape[0]
        r_i         = [None]*shape_vertices.shape[0]
        dot_prod_r  = [None]*shape_vertices.shape[0]

        # vector from x to vertex, shape=(...,dim) 
        n_vertices    = shape_vertices.shape[0]
        r_i[0]        = shape_vertices[0:1,:] - x
        norm_r_i[0]   = tr.linalg.norm(r_i[0], dim=1)

        # check if x is on an edge
        idx_on_edge = tr.isclose(norm_r_i[0], \
                         tr.zeros(1, dtype=self.dtype, device=self.device),\
                         rtol=1e-12, atol=1e-12)
        idx_stability = tr.logical_or(idx_stability, idx_on_edge)

        # -1 as tensor
        min_1 = -tr.ones(1, dtype=self.dtype, device=self.device)
        for idx, x_i in enumerate(shape_vertices[1:,:], 1):
            r_i[idx]        = x_i - x
            norm_r_i[idx]   = tr.linalg.norm(r_i[idx], dim=1)
            dot_prod_r[idx-1] = tr.matmul(r_i[idx-1].unsqueeze(1), \
                                          r_i[idx].unsqueeze(2))\
                                                        .squeeze(-1).squeeze(-1)

            # check at the same time the stability indeces
            # check if x is on an edge
            idx_on_edge = tr.isclose(norm_r_i[idx], \
                             tr.zeros(1, dtype=self.dtype, device=self.device),\
                             rtol=1e-12, atol=1e-12)
            idx_stability = tr.logical_or(idx_stability, idx_on_edge)

            # set zero to those values on the bdry
            idx_on_bdry= tr.isclose(dot_prod_r[idx-1].squeeze(-1).squeeze(-1)/\
                                 (norm_r_i[idx-1]*norm_r_i[idx]), \
                                         min_1, rtol=1e-12, atol=1e-12)
            idx_stability = tr.logical_or(idx_stability, idx_on_bdry)


        # last dot product
        dot_prod_r[-1] = tr.matmul(r_i[-1].unsqueeze(1), \
                                   r_i[0].unsqueeze(2))\
                                                    .squeeze(-1).squeeze(-1)

        # set zero to those values on the bdry
        idx_on_bdry= tr.isclose(dot_prod_r[-1].squeeze(-1).squeeze(-1)/\
                             (norm_r_i[-1]*norm_r_i[0]), \
                                     min_1, rtol=1e-12, atol=1e-12)
        idx_stability = tr.logical_or(idx_stability, idx_on_bdry)

        # first value of W
        W = 0.
        eval_idx = idx_stability == False
        for idx in range(n_vertices):
            r_i_vec   = r_i[idx][eval_idx, :]
            r_ip1_vec = r_i[(idx+1)%n_vertices][eval_idx,:]
            norm_r_i_vec    = norm_r_i[idx][eval_idx]
            norm_r_ip1_vec  = norm_r_i[(idx+1)%n_vertices][eval_idx]
            dot_prod_r_vec  = dot_prod_r[idx][eval_idx]

            # see (eq. 13), shape=(...,1)
            t_i   = tr.linalg.det(tr.cat((r_i_vec.unsqueeze(-1), \
                                          r_ip1_vec.unsqueeze(-1)), dim=2))/\
                    (  norm_r_i_vec*norm_r_ip1_vec + dot_prod_r_vec)

            # see (eq. 13), shape=(...,1)
            W     = W + (1./norm_r_i_vec + 1./norm_r_ip1_vec)*t_i


        # mean value potential
        Phi[eval_idx,0] = 2./W

        # reshape to the shape of the input x
        Phi = Phi.reshape(*x_shape_, 1)
        
        return Phi



# general elliptic test cases
class pde_general_elliptic_unitsq():
    def __init__(self, \
                 mesh_fenicsx=None, \
                 fem_sol_fenicsx=None, \
                 dtype=tr.double, \
                 device=tr.device('cpu')):
        # dtype and device agnostic code
        self.device = device
        self.dtype  = dtype

        # dimension 
        self.dim    = 2

        # parametric
        self.parametric = False

        # assign the fenicsx stuff
        self.msh_fnx       = mesh_fenicsx
        self.fem_sol_fnx   = fem_sol_fenicsx

        # vertices of L-shape in anti-clock-wise order, shape=(6,2)
        self.unitsq_vertices = tr.tensor([[0., 0.], [1., 0.], \
                                          [1., 1.], [0., 1.]], \
                                          dtype=self.dtype, device=self.device)

    def A(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim,dim)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        A = tr.zeros(*x_shape_, \
                     self.dim, self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

        # define the four subdomains
        idx_sub_omega    = [None]*4
        idx_sub_omega[0] =  (x[...,0] > 0.5)*(x[...,1] > 0.0)\
                           *(x[...,0] < 1.0)*(x[...,1] < 0.5)

        idx_sub_omega[1] =  (x[...,0] > 0.0)*(x[...,1] > 0.5)\
                           *(x[...,0] < 0.5)*(x[...,1] < 1.0)

        idx_sub_omega[2] =  (x[...,0] > 0.0)*(x[...,1] > 0.0)\
                           *(x[...,0] < 0.5)*(x[...,1] < 0.5)

        idx_sub_omega[3] =  (x[...,0] > 0.5)*(x[...,1] > 0.5)\
                           *(x[...,0] < 1.0)*(x[...,1] < 1.0)
        # subdomain 1
        A[..., 0, 0][idx_sub_omega[0]] = 2
        A[..., 1, 1][idx_sub_omega[0]] = 2
        A[..., 0, 1][idx_sub_omega[0]] = 1
        A[..., 1, 0][idx_sub_omega[0]] = 1
        # subdomain 2
        A[..., 0, 0][idx_sub_omega[1]] = 1/2
        A[..., 1, 1][idx_sub_omega[1]] = 1/2
        A[..., 0, 1][idx_sub_omega[1]] = 1/4
        A[..., 1, 0][idx_sub_omega[1]] = 1/4
        # subdomain 3
        A[..., 0, 0][idx_sub_omega[2]] = 1
        A[..., 1, 1][idx_sub_omega[2]] = 1
        A[..., 0, 1][idx_sub_omega[2]] = 1/2
        A[..., 1, 0][idx_sub_omega[2]] = 1/2
        # subdomain 4
        A[..., 0, 0][idx_sub_omega[3]] = 1
        A[..., 1, 1][idx_sub_omega[3]] = 1
        A[..., 0, 1][idx_sub_omega[3]] = 1/2
        A[..., 1, 0][idx_sub_omega[3]] = 1/2


        return A 

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


    def b(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        b = tr.zeros(*x_shape_, \
                     self.dim, \
                     dtype=self.dtype, \
                     device=self.device)
        b[...,0] = 1.

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

        # calc the function 
        result[..., 0] = 10*tr.abs(tr.sin(2*pi*x[...,0])*tr.sin(2*pi*x[...,1]))

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


    def u_sol(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # from torch to numpy
        x_np = np.zeros((3, x.shape[0]))
        x_np[:2,:] = x.detach().cpu().transpose(0,1)

        # shape (dim,...)
        u_tr = tr.tensor(self.u_sol_fenicsx(x_np), \
                         dtype=self.dtype, \
                         device=self.device).transpose(0,1)

        return u_tr


    def char_func_omega(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        
        # get the indices on unit square
        idx_square =    (x[...,0] > 0. )*(x[...,1] > 0.0)\
                       *(x[...,0] < 1.0)*(x[...,1] < 1.0)
        
        # evaluate the characteristic function
        value_tr  = tr.zeros(*list(x.shape[:-1]), \
                             1, \
                             dtype=self.dtype,\
                             device=self.device)

        value_tr[...,0][idx_square] = 1.

        return value_tr


    ### Fenicsx methods
    def A_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim*dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        A_tr = self.A(x_tr)

        # reshape and torch into numpy
        A_np = tr.transpose(tr.flatten(A_tr, start_dim = 1), 0, 1).detach()\
                                                                  .numpy()

        return A_np


    def b_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        b_tr = self.b(x_tr)

        # reshape and torch into numpy
        b_np = tr.transpose(b_tr, 0, 1).detach().numpy()

        return b_np


    def c_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        c_tr = self.c(x_tr)

        # reshape and torch into numpy
        c_np = tr.transpose(c_tr, 0, 1).detach().numpy()

        return c_np


    def f_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        f_tr = self.f(x_tr)
        f_np = tr.transpose(f_tr,0,1).detach().numpy()

        return f_np


    def mean_value_potential_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        mvp_tr = self.mean_value_potential(x_tr)
        mvp_np = tr.transpose(mvp_tr,0,1).detach().numpy()

        return mvp_np


    def uD_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        uD_tr = self.uD(x_tr)

        # reshape and torch into numpy
        uD_np = tr.transpose(uD_tr, 0, 1).detach().numpy()

        return uD_np


    def u_sol_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)

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
        for i, point in enumerate(x_np):
            if len(colliding_cells.links(i)) > 0:
                actual_cells.append(colliding_cells.links(i)[0])
            else:
                ValueError("Point "+repr(i)+" is not inside the domain")

        u_sol_np = np.transpose(self.fem_sol_fnx.eval(x_np, actual_cells))

        return u_sol_np


    def char_func_omega_fenicsx(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        x_tr     = tr.transpose(tr.tensor(x[0:2,:], dtype=tr.double), 0, 1)
        
        # evaluate the char function
        value_np = tr.transpose(self.char_func_omega(x_tr),0,1).detach().numpy()
        
        return value_np


    def coercivity_const(self):
        # Output: Coercivity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the coercivity constant (minimal eigenvalue of diff. matrix)

        return 1./4


    def mean_value_potential(self, x):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        # Implements the approximate distance function to impose dirichlet 
        # bdry condtions exactly. Reference:
        # [Exact imposition of boundary conditions with distance functions in
        #  physics-informed deep neural networks]

        # x shape
        x_shape_ = list(x.shape[:-1])

        # in case the pde is parametric
        x = x[...,:self.dim].reshape(-1,self.dim)

        # mean value potentials (eq. 14)
        Phi = tr.zeros(x.shape[0], \
                       1, \
                       dtype=self.dtype, \
                       device=self.device)

        # eval only points within the domain, zero otherwise
        idx_stability = tr.logical_not((x[...,1] > 0.)*(x[...,1] < 1.)\
                                      *(x[...,0] > 0.)*(x[...,0] < 1.))
 
        # calculate first all necessary quantities
        # data structures
        norm_r_i    = [None]*self.unitsq_vertices.shape[0]
        r_i         = [None]*self.unitsq_vertices.shape[0]
        dot_prod_r  = [None]*self.unitsq_vertices.shape[0]

        # vector from x to vertex, shape=(...,dim) 
        n_vertices = self.unitsq_vertices.shape[0]
        r_i[0]        = self.unitsq_vertices[0:1,:] - x
        norm_r_i[0]   = tr.linalg.norm(r_i[0], dim=1)

        # check if x is on an edge
        idx_on_edge = tr.isclose(norm_r_i[0], \
                         tr.zeros(1, dtype=self.dtype, device=self.device),\
                         rtol=1e-12, atol=1e-12)
        idx_stability = tr.logical_or(idx_stability, idx_on_edge)

        # -1 as tensor
        min_1 = -tr.ones(1, dtype=self.dtype, device=self.device)
        for idx, x_i in enumerate(self.unitsq_vertices[1:,:], 1):
            r_i[idx]        = x_i - x
            norm_r_i[idx]   = tr.linalg.norm(r_i[idx], dim=1)
            dot_prod_r[idx-1] = tr.matmul(r_i[idx-1].unsqueeze(1), \
                                          r_i[idx].unsqueeze(2))\
                                                        .squeeze(-1).squeeze(-1)

            # check at the same time the stability indeces
            # check if x is on an edge
            idx_on_edge = tr.isclose(norm_r_i[idx], \
                             tr.zeros(1, dtype=self.dtype, device=self.device),\
                             rtol=1e-12, atol=1e-12)
            idx_stability = tr.logical_or(idx_stability, idx_on_edge)

            # set zero to those values on the bdry
            idx_on_bdry= tr.isclose(dot_prod_r[idx-1].squeeze(-1).squeeze(-1)/\
                                 (norm_r_i[idx-1]*norm_r_i[idx]), \
                                         min_1, rtol=1e-12, atol=1e-12)
            idx_stability = tr.logical_or(idx_stability, idx_on_bdry)


        # last dot product
        dot_prod_r[-1] = tr.matmul(r_i[-1].unsqueeze(1), \
                                   r_i[0].unsqueeze(2))\
                                                    .squeeze(-1).squeeze(-1)

        # set zero to those values on the bdry
        idx_on_bdry= tr.isclose(dot_prod_r[-1].squeeze(-1).squeeze(-1)/\
                             (norm_r_i[-1]*norm_r_i[0]), \
                                     min_1, rtol=1e-12, atol=1e-12)
        idx_stability = tr.logical_or(idx_stability, idx_on_bdry)

        # first value of W
        W = 0.
        eval_idx = idx_stability == False
        for idx in range(n_vertices):
            r_i_vec   = r_i[idx][eval_idx, :]
            r_ip1_vec = r_i[(idx+1)%n_vertices][eval_idx,:]
            norm_r_i_vec    = norm_r_i[idx][eval_idx]
            norm_r_ip1_vec  = norm_r_i[(idx+1)%n_vertices][eval_idx]
            dot_prod_r_vec  = dot_prod_r[idx][eval_idx]

            # see (eq. 13), shape=(...,1)
            t_i   = tr.linalg.det(tr.cat((r_i_vec.unsqueeze(-1), \
                                          r_ip1_vec.unsqueeze(-1)), dim=2))/\
                    (  norm_r_i_vec*norm_r_ip1_vec + dot_prod_r_vec)

            # see (eq. 13), shape=(...,1)
            W     = W + (1./norm_r_i_vec + 1./norm_r_ip1_vec)*t_i


        # mean value potential
        Phi[eval_idx,0] = 2./W

        # reshape to the shape of the input x
        Phi = Phi.reshape(*x_shape_, 1)
        
        return Phi



# general elliptic test cases
class pde_general_elliptic_unitcircle():
    def __init__(self, \
                 mesh_fenicsx=None, \
                 fem_sol_fenicsx=None, \
                 dtype=tr.double, \
                 device=tr.device('cpu')):
        # dtype and device agnostic code
        self.device = device
        self.dtype  = dtype

        # dimension 
        self.dim    = 2

        # parametric
        self.parametric = False

        # assign the fenicsx stuff
        self.msh_fnx       = mesh_fenicsx
        self.fem_sol_fnx   = fem_sol_fenicsx


    def A(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim,dim)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        A = tr.zeros(*x_shape_, \
                     self.dim, self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

        # define the four subdomains
        idx_sub_omega    = [None]*4
        idx_sub_omega[0] =  (x[...,0] > 0.5)*(x[...,1] > 0.0)\
                           *(x[...,0] < 1.0)*(x[...,1] < 0.5)

        idx_sub_omega[1] =  (x[...,0] > 0.0)*(x[...,1] > 0.5)\
                           *(x[...,0] < 0.5)*(x[...,1] < 1.0)

        idx_sub_omega[2] =  (x[...,0] > 0.0)*(x[...,1] > 0.0)\
                           *(x[...,0] < 0.5)*(x[...,1] < 0.5)

        idx_sub_omega[3] =  (x[...,0] > 0.5)*(x[...,1] > 0.5)\
                           *(x[...,0] < 1.0)*(x[...,1] < 1.0)
        # subdomain 1
        A[..., 0, 0][idx_sub_omega[0]] = 2
        A[..., 1, 1][idx_sub_omega[0]] = 2
        A[..., 0, 1][idx_sub_omega[0]] = 1
        A[..., 1, 0][idx_sub_omega[0]] = 1
        # subdomain 2
        A[..., 0, 0][idx_sub_omega[1]] = 1/2
        A[..., 1, 1][idx_sub_omega[1]] = 1/2
        A[..., 0, 1][idx_sub_omega[1]] = 1/4
        A[..., 1, 0][idx_sub_omega[1]] = 1/4
        # subdomain 3
        A[..., 0, 0][idx_sub_omega[2]] = 1
        A[..., 1, 1][idx_sub_omega[2]] = 1
        A[..., 0, 1][idx_sub_omega[2]] = 1/2
        A[..., 1, 0][idx_sub_omega[2]] = 1/2
        # subdomain 4
        A[..., 0, 0][idx_sub_omega[3]] = 1
        A[..., 1, 1][idx_sub_omega[3]] = 1
        A[..., 0, 1][idx_sub_omega[3]] = 1/2
        A[..., 1, 0][idx_sub_omega[3]] = 1/2


        return A 

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


    def b(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        b = tr.zeros(*x_shape_, \
                     self.dim, \
                     dtype=self.dtype, \
                     device=self.device)
        b[...,0] = 1.

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

        # calc the function 
        result[..., 0] =  x[..., 0]**2*x[..., 1]**2 + x[...,1]**3 + 3

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


    def u_sol(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # from torch to numpy
        x_np = np.zeros((3, x.shape[0]))
        x_np[:2,:] = x.detach().cpu().transpose(0,1)

        # shape (dim,...)
        u_tr = tr.tensor(self.u_sol_fenicsx(x_np), \
                         dtype=self.dtype, \
                         device=self.device).transpose(0,1)

        return u_tr


    def char_func_omega(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        
        # get the indices on unit square
        idx_square =  ((x[...,0]-0.5)**2 + (x[...,1]-0.5)**2) < 0.5**2
        
        # evaluate the characteristic function
        value_tr  = tr.zeros(*list(x.shape[:-1]), \
                             1, \
                             dtype=self.dtype,\
                             device=self.device)

        value_tr[...,0][idx_square] = 1.

        return value_tr


    ### Fenicsx methods
    def A_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim*dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        A_tr = self.A(x_tr)

        # reshape and torch into numpy
        A_np = tr.transpose(tr.flatten(A_tr, start_dim = 1), 0, 1).detach()\
                                                                  .numpy()

        return A_np


    def b_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        b_tr = self.b(x_tr)

        # reshape and torch into numpy
        b_np = tr.transpose(b_tr, 0, 1).detach().numpy()

        return b_np


    def c_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        c_tr = self.c(x_tr)

        # reshape and torch into numpy
        c_np = tr.transpose(c_tr, 0, 1).detach().numpy()

        return c_np


    def f_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        f_tr = self.f(x_tr)
        f_np = tr.transpose(f_tr,0,1).detach().numpy()

        return f_np


    def mean_value_potential_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        mvp_tr = self.mean_value_potential(x_tr)
        mvp_np = tr.transpose(mvp_tr,0,1).detach().numpy()

        return mvp_np


    def uD_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        uD_tr = self.uD(x_tr)

        # reshape and torch into numpy
        uD_np = tr.transpose(uD_tr, 0, 1).detach().numpy()

        return uD_np


    def u_sol_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)

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
        for i, point in enumerate(x_np):
            if len(colliding_cells.links(i)) > 0:
                actual_cells.append(colliding_cells.links(i)[0])
            else:
                ValueError("Point "+repr(i)+" is not inside the domain")

        u_sol_np = np.transpose(self.fem_sol_fnx.eval(x_np, actual_cells))

        return u_sol_np


    def char_func_omega_fenicsx(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        x_tr     = tr.transpose(tr.tensor(x[0:2,:], dtype=tr.double), 0, 1)
        
        # evaluate the char function
        value_np = tr.transpose(self.char_func_omega(x_tr),0,1).detach().numpy()
        
        return value_np


    def coercivity_const(self):
        # Output: Coercivity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the coercivity constant (minimal eigenvalue of diff. matrix)

        return 1./4


    def mean_value_potential(self, x):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        # Implements the approximate distance function to impose dirichlet 
        # bdry condtions exactly. Reference:
        # [Exact imposition of boundary conditions with distance functions in
        #  physics-informed deep neural networks]
        # eq. (7)

        # x shape
        x_shape_ = list(x.shape[:-1])

        # in case the pde is parametric
        x = x[...,:self.dim].reshape(-1,self.dim)

        # mean value potentials (eq. 14)
        radi = 0.5
        Phi  = (radi**2 - tr.matmul((x.reshape(-1,1,self.dim)-0.5), \
                                    (x.reshape(-1,self.dim,1)-0.5))\
                                                 .reshape(-1,1))/(2*radi)

       
        return Phi


# general elliptic test cases
class pde_general_elliptic_unitdonut():
    def __init__(self, \
                 mesh_fenicsx=None, \
                 fem_sol_fenicsx=None, \
                 dtype=tr.double, \
                 device=tr.device('cpu')):
        # dtype and device agnostic code
        self.device = device
        self.dtype  = dtype

        # dimension 
        self.dim    = 2

        # parametric
        self.parametric = False

        # assign the fenicsx stuff
        self.msh_fnx       = mesh_fenicsx
        self.fem_sol_fnx   = fem_sol_fenicsx


    def A(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim,dim)
        
        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        A = tr.zeros(*x_shape_, \
                     self.dim, self.dim, \
                     dtype=self.dtype, \
                     device=self.device)

        # define the four subdomains
        idx_sub_omega    = [None]*4
        idx_sub_omega[0] =  (x[...,0] > 0.5)*(x[...,1] > 0.0)\
                           *(x[...,0] < 1.0)*(x[...,1] < 0.5)

        idx_sub_omega[1] =  (x[...,0] > 0.0)*(x[...,1] > 0.5)\
                           *(x[...,0] < 0.5)*(x[...,1] < 1.0)

        idx_sub_omega[2] =  (x[...,0] > 0.0)*(x[...,1] > 0.0)\
                           *(x[...,0] < 0.5)*(x[...,1] < 0.5)

        idx_sub_omega[3] =  (x[...,0] > 0.5)*(x[...,1] > 0.5)\
                           *(x[...,0] < 1.0)*(x[...,1] < 1.0)
        # subdomain 1
        A[..., 0, 0][idx_sub_omega[0]] = 2
        A[..., 1, 1][idx_sub_omega[0]] = 2
        A[..., 0, 1][idx_sub_omega[0]] = 1
        A[..., 1, 0][idx_sub_omega[0]] = 1
        # subdomain 2
        A[..., 0, 0][idx_sub_omega[1]] = 1/2
        A[..., 1, 1][idx_sub_omega[1]] = 1/2
        A[..., 0, 1][idx_sub_omega[1]] = 1/4
        A[..., 1, 0][idx_sub_omega[1]] = 1/4
        # subdomain 3
        A[..., 0, 0][idx_sub_omega[2]] = 1
        A[..., 1, 1][idx_sub_omega[2]] = 1
        A[..., 0, 1][idx_sub_omega[2]] = 1/2
        A[..., 1, 0][idx_sub_omega[2]] = 1/2
        # subdomain 4
        A[..., 0, 0][idx_sub_omega[3]] = 1
        A[..., 1, 1][idx_sub_omega[3]] = 1
        A[..., 0, 1][idx_sub_omega[3]] = 1/2
        A[..., 1, 0][idx_sub_omega[3]] = 1/2


        return A 

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


    def b(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # x shape
        x_shape_ = list(x.shape[:-1])

        # data structure
        b = tr.zeros(*x_shape_, \
                     self.dim, \
                     dtype=self.dtype, \
                     device=self.device)
        b[...,0] = 1.

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

        # calc the function 
        result[..., 0] =  2*x[..., 0]**2*x[..., 1]**2 + 10*x[...,1]**3 + 3

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


    def u_sol(self, x, **kwargs):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (...,dim)

        # from torch to numpy
        x_np = np.zeros((3, x.shape[0]))
        x_np[:2,:] = x.detach().cpu().transpose(0,1)

        # shape (dim,...)
        u_tr = tr.tensor(self.u_sol_fenicsx(x_np), \
                         dtype=self.dtype, \
                         device=self.device).transpose(0,1)

        return u_tr


    def char_func_omega(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        
        # get the indices on unit square
        idx_circle_out =  ((x[...,0]-0.5)**2 + (x[...,1]-0.5)**2) < 0.5**2
        idx_circle_in  =  ((x[...,0]-0.5)**2 + (x[...,1]-0.5)**2) < 0.25**2
        idx_donut      = tr.logical_and(idx_circle_out, \
                                        tr.logical_not(idx_circle_in))
        
        # evaluate the characteristic function
        value_tr  = tr.zeros(*list(x.shape[:-1]), \
                             1, \
                             dtype=self.dtype,\
                             device=self.device)

        value_tr[...,0][idx_donut] = 1.

        return value_tr


    ### Fenicsx methods
    def A_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim*dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        A_tr = self.A(x_tr)

        # reshape and torch into numpy
        A_np = tr.transpose(tr.flatten(A_tr, start_dim = 1), 0, 1).detach()\
                                                                  .numpy()

        return A_np


    def b_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        b_tr = self.b(x_tr)

        # reshape and torch into numpy
        b_np = tr.transpose(b_tr, 0, 1).detach().numpy()

        return b_np


    def c_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        c_tr = self.c(x_tr)

        # reshape and torch into numpy
        c_np = tr.transpose(c_tr, 0, 1).detach().numpy()

        return c_np


    def f_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        f_tr = self.f(x_tr)
        f_np = tr.transpose(f_tr,0,1).detach().numpy()

        return f_np


    def mean_value_potential_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.transpose(tr.tensor(x[:2,:], dtype=self.dtype), 0, 1)

        ## evaluate f
        mvp_tr = self.mean_value_potential(x_tr)
        mvp_np = tr.transpose(mvp_tr,0,1).detach().numpy()

        return mvp_np


    def uD_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)
        
        # numpy into torch: shape = (...,dim)
        x_tr = tr.tensor(x[:2,:], dtype=self.dtype).transpose(0, 1)

        # call method
        uD_tr = self.uD(x_tr)

        # reshape and torch into numpy
        uD_np = tr.transpose(uD_tr, 0, 1).detach().numpy()

        return uD_np


    def u_sol_fenicsx(self, x, **kwargs):
        # Input: x.shape = (dim,...), (numpy array)
        # Output: out.shape = (dim,...), (numpy array)

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
        for i, point in enumerate(x_np):
            if len(colliding_cells.links(i)) > 0:
                actual_cells.append(colliding_cells.links(i)[0])
            else:
                ValueError("Point "+repr(i)+" is not inside the domain")

        u_sol_np = np.transpose(self.fem_sol_fnx.eval(x_np, actual_cells))

        return u_sol_np


    def char_func_omega_fenicsx(self, x, **kwargs):
        # returns 1 for x \in \Omega=L-shape and zero else
        x_tr     = tr.transpose(tr.tensor(x[0:2,:], dtype=tr.double), 0, 1)
        
        # evaluate the char function
        value_np = tr.transpose(self.char_func_omega(x_tr),0,1).detach().numpy()
        
        return value_np


    def coercivity_const(self):
        # Output: Coercivity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the coercivity constant (minimal eigenvalue of diff. matrix)

        return 1./4


    def mean_value_potential(self, x):
        # Input: x.shape = (...,dim)
        # Output: out.shape = (..,1)
        # Implements the approximate distance function to impose dirichlet 
        # bdry condtions exactly. Reference:
        # [Exact imposition of boundary conditions with distance functions in
        #  physics-informed deep neural networks]
        # eq. (7)

        # x shape
        x_shape_ = list(x.shape[:-1])

        # in case the pde is parametric
        x = x[...,:self.dim].reshape(-1,self.dim)

        # mean value potentials (eq. 14)
        # outer circle
        radi = 0.5
        Phi_1  = (radi**2 - tr.matmul((x.reshape(-1,1,self.dim)-0.5), \
                                      (x.reshape(-1,self.dim,1)-0.5))\
                                                 .reshape(-1,1))/(2*radi)
        # inner circle
        radi = 0.25
        Phi_2  = (radi**2 - tr.matmul((x.reshape(-1,1,self.dim)-0.5), \
                                      (x.reshape(-1,self.dim,1)-0.5))\
                                                 .reshape(-1,1))/(2*radi)

      
        return Phi_1*Phi_2



# general elliptic test cases
class pde_general_elliptic_arkansas():
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
        self.dim = 2

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

        A_diff[...,0,0] = 2
        A_diff[...,1,1] = 1

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

        b[...,0] = -2*tr.sin(mu*x[...,1])
        b[...,1] = 2*tr.cos(mu*x[...,0])

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
        result[..., 0] = 1

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


    def coercivity_const(self, mu):
        # Output: Coercivity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the coercivity constant (minimal eigenvalue of diff. matrix)

        return 1.


    def continuity_const(self, mu):
        # Output: Continuity constant w.r.t. H1-Semi-norm, out.shape = (1,)

        # return the continuity constant (maximal eigenvalue of diff. matrix)
        return 6.


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

        # reshape to the shape of the input x
        Phi = Phi.reshape(*x_shape_, 1)
        return Phi


