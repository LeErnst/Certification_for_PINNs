import torch as tr
import numpy as np
import sys
import math
from math import log2, ceil, factorial, floor, pi, sqrt
import copy
from ..wavelets.rl_bior_scal_mask import wname_to_code
from .inner_products import kron_nD, torch_cartesian_nD

################################ auxilary functions ############################

# check if real number x is a dyadic point and if so return the level
def get_level(x):
    x = abs(x)
    x = x-floor(x)
    j = 1
    if abs(x) < 1e-15 or abs(x) != x:
        return 0

    while True:
        x *= 2
        if x > 1:
            x = x-1
        elif abs(1-x) < 1e-15:
            break
        elif j > 200:
            return -1
        j += 1

    return j



################# class for quadrature data in n dimensions ####################
class quadrature_data():
    def __init__(self, \
                 dim,\
                 weights_list, \
                 points_list, \
                 Linear_Transform,\
                 DetJacLinear_Transform,\
                 dtype=tr.double, \
                 device=tr.device('cpu')):

        # dtype and device agnostic code
        self.dtype  = dtype
        self.device = device

        # assign important data
        # dimension
        self.dim = dim

        # check if data are lists
        if not isinstance(weights_list, list):
            weights_list = [weights_list]
        if not isinstance(points_list, list):
            points_list = [points_list]

        # length of points and weights must fit
        assert len(weights_list) == len(points_list)

        # if dim is different but weights_list has length 1, given weights in 
        # all dimensions
        if self.dim != len(weights_list):
            if len(weights_list) == 1:
                weights_list = weights_list*self.dim
                points_list  = points_list*self.dim
            else:
                raise ValueError("Weights and dimension must fit")


        # quadrature weights and support points
        self.weights    = weights_list
        self.weigths_nD = kron_nD(self.weights)
        self.points     = points_list
        self.points_nD  = torch_cartesian_nD(self.points)

        # assign the linear transformation
        self.F_transf     = Linear_Transform
        self.detJF_transf = DetJacLinear_Transform


#################### domain discretizations and affine transformations #########

### K_hat = (-1,1)**d ; K = \prod_{i=1...d}([a_i,b_i]) (cartesian product)
### F_K : K_hat -> K
def F_K(x_hat, P1, P2):
    return tr.matmul(tr.diag((1/2*(P2-P1)).flatten()), x_hat)+1/2*(P1+P2)

### F_K^-1 : K -> K_hat
def F_K_1(x, P1, P2):
    B_1 = tr.diag(1/(P2-P1).flatten())
    return 2*tr.matmul(B_1, x_hat) - tr.matmul(B_1, (P1+P2))

### det(JF_K) = det(B)
def detJF_K(P1, P2):
    return tr.prod((1/2*(P2-P1)).flatten())

### B^-1
def B_1(P1, P2):
    return 2*tr.diag((1/(P2-P1)).flatten())


###### vectorized variants for GPU implementation
###### K_hat = (-1,1)**d
## F_K : K_hat -> K
def F_K_v(x_hat, P1, P2):
    # Input: P1.shape = (N,d), P2.shape = (N,d), x_hat.shape = (n,d)
    # N: number of elements
    # n: number of support vectors of quadrature rule for nD
    # Output: X.shape = (N,n,d)
    return tr.add(tr.mul(tr.mul(0.5,tr.sub(P2,P1)).unsqueeze(1), \
                         x_hat.unsqueeze(0)) , \
                  tr.mul(0.5,tr.add(P1,P2)).unsqueeze(1))


### det(JF_K) = det(B)
def detJF_K_v(P1, P2):
    # Input: P1.shape = (N,d), P2.shape = (N,d)
    # N: number of elements
    # Output: out.shape = (N,1)
    return tr.prod(tr.mul(0.5,tr.sub(P2,P1)), \
                   dim=1, \
                   keepdim=True).reshape(-1,1)

### B^-1
def B_1_v(P1, P2):
    # Input: P1.shape = (N,d), P2.shape = (N,d)
    # N: number of elements
    # Output: out.shape = (N,1,d) (actually (N,d,d) but zeros are not stored, 
    #         only the diagonal)
    N = P1.shape[0]
    return tr.div(2.,tr.sub(P2,P1)).reshape(N,1,2)


### mapping from [-1,1] -> rect(P1,P2) for line integrals in 2D
# TODO: consider generalization to [-1,1]**(d-1) -> rect(P1,P2)
def f_k_v(s_hat, P1, P2):
    # Input: P1.shape = (N,2), P2.shape = (N,2), x_hat.shape = (n,)
    # N: number of lines
    # n: number of support vectors of quadrature rule for 1D
    # Output: X.shape = (N,n,2)
    N = P1.shape[0]
    n = s_hat.shape[0]
    term1 = tr.div(tr.add(1.,s_hat),2.).reshape(n,1)
    return tr.mul(P2.reshape(N,1,2),term1) + \
           tr.mul(P1.reshape(N,1,2),tr.sub(1,term1))

### det of mapping from [-1,1] -> rect(P1,P2) for line integrals in 2D
# TODO: consider generalization to [-1,1]**(d-1) -> rect(P1,P2)
def detJf_k_v(P1, P2):
    # Input: P1.shape = (N,2), P2.shape = (N,2)
    # N: number of lines
    # n: number of support vectors of quadrature rule for 1D
    # Output: out.shape = (N,1,1)
    return tr.div(tr.norm(tr.sub(P2,P1),dim=1),2.).reshape(-1,1,1)


###### K_hat = (0,1)**2
## F_K : K_hat -> K
def F_K_01(x_hat, P1, P2):
    # Input: P1.shape = (N,2), P2.shape = (N,2), x_hat.shape = (n,2)
    # N: number of elements
    # n: number of support vectors of quadrature rule for 1D
    # Output: X.shape = (N,n,2)
    N = P1.shape[0]
    n = x_hat.shape[0]
    return tr.add(tr.mul(tr.sub(P2,P1).reshape(N,1,2), \
                         x_hat.reshape(1,n,2)), \
                  P1.reshape(N,1,2))

### det(JF_K) = det(B)
def detJF_K_01(P1, P2):
    # Input: P1.shape = (N,2), P2.shape = (N,2)
    # N: number of elements
    # Output: out.shape = (N,1,1)
    N = P1.shape[0]
    return tr.prod(tr.sub(P2,P1), \
                   dim=1, \
                   keepdim=True).reshape(N,1,1)

### B^-1
def B_1_01(P1, P2):
    # Input: P1.shape = (N,2), P2.shape = (N,2)
    # N: number of elements
    # Output: out.shape = (N,1,2) (actually (N,2,2) but zeros are not stored)
    N = P1.shape[0]
    return tr.div(1.,tr.sub(P2,P1)).reshape(N,1,2)


### mapping from [0,1] -> conv(P1,P2) for line integrals
def f_k_01(s_hat, P1, P2):
    # Input: P1.shape = (N,2), P2.shape = (N,2), s_hat.shape = (n,)
    # N: number of lines
    # n: number of support vectors of quadrature rule for 1D
    # Output: X.shape = (N,n,2)
    N = P1.shape[0]
    n = s_hat.shape[0]
    return tr.mul(P2.reshape(N,1,2),s_hat.reshape(n,1)) + \
           tr.mul(P1.reshape(N,1,2),tr.sub(1,s_hat.reshape(n,1)))

def detJf_k_01(P1, P2):
    # Input: P1.shape = (N,2), P2.shape = (N,2)
    # N: number of lines
    # n: number of support vectors of quadrature rule for 1D
    # Output: out.shape = (N,1,1)
    return tr.norm(tr.sub(P2,P1),dim=1).reshape(-1,1,1)


####### Discretization of an interval (a,b)
class disc_interval():
    def __init__(self,
                 a,
                 b,
                 N,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        # data type
        self.dtype = dtype
        # device
        self.device = device

        # interval [a,b]
        self.a = a
        self.b = b

        # number of coordinates including the ones at a and b
        self.numco = N
        # number of elements
        self.numel = self.numco - 1

        # discretization
        # coordinates
        self.co = tr.linspace(a, b, self.numco, \
                              dtype=self.dtype, \
                              device=self.device).reshape(self.numco,1)

        # elements
        self.el = tr.empty(self.numel, 2, dtype=tr.long, device=self.device)

        for i in range(self.numel):
            self.el[i,0] = i
            self.el[i,1] = i + 1

        # bdry
        self.bdry = tr.tensor([0, self.numco-1], \
                              dtype=tr.long, \
                              device=self.device)



############## coordinates, elements (for Omega=(0,1)^2)
### discretization class, which discretize Omega with axis parallel cubes
class disc_unit_square_2d():
    # discretize Omega=(0,1)^2 with h^2-squares
    def __init__(self, 
                 h, 
                 dtype=tr.double, 
                 device="cpu"):
        self.device  = device
        self.dtype   = dtype
        self.h       = h
        self.N       = ceil(1/h)+1
        self.numco   = self.N**2
        self.numel   = (self.N-1)**2
        self.numbdry = (self.N-1)*4
        self.numedge = 2*self.N*(self.N-1)
        self.co      = tr.empty(self.numco  , 2, dtype=dtype  , device=device)
        self.el      = tr.empty(self.numel  , 4, dtype=tr.long, device=device)
        self.bdry    = tr.empty(self.numbdry, 2, dtype=tr.long, device=device)
        self.edge    = tr.empty(self.numedge, 2, dtype=tr.long, device=device)
        self.el2edge = tr.empty(self.numel  , 4, dtype=tr.long, device=device)

        ### fill the data structures
        #### coordinates
        x = tr.linspace(0, 1, self.N, dtype=dtype)
        y = tr.linspace(0, 1, self.N, dtype=dtype)
        X, Y = tr.meshgrid(x,y)
        # column wise through X and Y
        for j in range(self.N):
            for i in range(self.N):
                self.co[self.N*j + i, 0] = X[i,j]
                self.co[self.N*j + i, 1] = Y[i,j]

        #### elements
        # N+1 N+2 N+3 N+4 N+5     2*N
        #  +---+---+---+---+   +---+
        #  | 1 | 2 | 3 | 4 |...| k |
        #  +---+---+---+---+   +---+
        #  1   2   3   4   5  N-1  N
        #
        for i in range(self.N-1):
            for j in range(self.N-1):
                # anti-clockwise
                self.el[(self.N-1)*i+j,0] = self.N*i + j
                self.el[(self.N-1)*i+j,1] = self.N*i + j + 1
                self.el[(self.N-1)*i+j,2] = self.N*(i+1) + j + 1
                self.el[(self.N-1)*i+j,3] = self.N*(i+1) + j

        #### edges
        #  +---+---+---+---+      +-----+
        #  |   |   |   |   |      |     |
        #  N  N+1 N+2 N+3 N+4....2N-1  2N
        #  |   |   |   |   |      |     |
        #  +-1-+-2-+-3-+-4-+      +-N-1-+
        #
        for i in range(self.N-1):
            for j in range(self.N-1):
                # horizontal edges
                self.edge[(2*self.N-1)*i+j,:] = tr.tensor([self.N*i+j,\
                                                           self.N*i+j+1])
                # vertical edges
                self.edge[(2*self.N-1)*i+self.N-1+j,:] = \
                                          tr.tensor([self.N*i+j,self.N*(i+1)+j])

            # last vertical edge
            self.edge[(2*self.N-1)*i+self.N-1+self.N-1,:] = \
                            tr.tensor([self.N*i+self.N-1,self.N*(i+1)+self.N-1])
        
        # upper horizontal edges
        for i in range(self.N-1):
            self.edge[self.numedge-self.N+1+i,:] = \
                            tr.tensor([self.N**2-self.N+i,self.N**2-self.N+i+1])
                                                              

        #### elements 2 edges
        # anti-clockwise 
        for i in range(self.N-1):
            for j in range(self.N-1):
                self.el2edge[(self.N-1)*i+j,0] = (2*self.N-1)*i+j
                self.el2edge[(self.N-1)*i+j,1] = (2*self.N-1)*i+self.N-1+j
                self.el2edge[(self.N-1)*i+j,2] = (2*self.N-1)*(i+1)+j
                self.el2edge[(self.N-1)*i+j,3] = (2*self.N-1)*i+self.N-1+j+1

        #### boundary
        # element, edge
        for i in range(self.N-1):
            # left
            self.bdry[3*(self.N-1)+i] = tr.tensor([(self.N-1)*i,\
                                                   (2*self.N-1)*i+self.N-1])
            # right 
            self.bdry[(self.N-1)+i] = tr.tensor([(self.N-1)*i+(self.N-2),\
                                              (2*self.N-1)*i+self.N-1+self.N-1])
        # bottom
        self.bdry[0:self.N-1,0] = tr.arange(0,self.N-1)
        self.bdry[0:self.N-1,1] = tr.arange(0,self.N-1)
        # top
        self.bdry[2*(self.N-1):3*(self.N-1),0] = tr.arange(self.numel-self.N+1,\
                                                           self.numel)
        self.bdry[2*(self.N-1):3*(self.N-1),1] = tr.arange(\
                                                         self.numedge-self.N+1,\
                                                         self.numedge)
                                                   


    # method to calculate the outward normals from self.bdry
    def outward_normal(self, el_edge):
        # Input: el_edge: shape = (n,2), element number - edge number
        #                 combinations
        # Output: n: shape = (n, 1, 2)

        # mid point coordinates of elements
        # shape = (n, 2)
        mid = tr.div(tr.sum(self.co[self.el[el_edge[:,0]]],dim=1),\
                     self.el.shape[1])
        # bdry node first coordinates
        # shape = (n, 2)
        P1 = self.co[self.edge[el_edge[:,1],0],:]
        # bdry node second coordinates
        # shape = (n, 2)
        P2 = self.co[self.edge[el_edge[:,1],1],:]
        # outward normal
        # shape = (n, 1, 2)
        n = tr.sub(tr.mul(0.5,tr.add(P1,P2)),mid)
        n = tr.div(n,tr.norm(n,dim=1).reshape(n.shape[0],1))

        return n.reshape(el_edge.shape[0], 1, 2)


### discretization class, which discretizes Omega with uniform grid for splines  
class uniform_unit_square_Bsplines():
    def __init__(self, 
                 N=[10, 10],
                 dtype=tr.double,
                 device=tr.device('cpu')):
        # specialized for spline usage
        # Input: dim: list of number of knots per dimension,
        #             e.g. [N_time, N_space_1, N_space_2,...]

        # data type and device
        self.dtype  = dtype
        self.device = device
        # unit square dimension
        self.d  = len(N)
        # number of knots
        self.N  = N
        # knot coordinates, a tensor product of all entries in the list defines
        # the grid, which we do not need to store explicitly
        self.co = [tr.linspace(0, 1, N[i], \
                               dtype=self.dtype, \
                               device=self.device) for i in range(self.d)]


    def get_el(self):
        # get the element defining knots in a specific order
        P1 = tr.cartesian_prod(*[self.co[i][ :-1] for i in range(self.d)])
        P2 = tr.cartesian_prod(*[self.co[i][1:  ] for i in range(self.d)])
        if self.d == 1:
            P1 = tr.unsqueeze(P1, dim=1)
            P2 = tr.unsqueeze(P2, dim=1)

        return P1, P2

    def get_initial(self):
        # get the initial element defining knots in a specific order
        P1 = tr.cartesian_prod(*[self.co[i][ :-1] for i in range(1,self.d)])
        P2 = tr.cartesian_prod(*[self.co[i][1:  ] for i in range(1,self.d)])
        if self.d == 2:
            P1 = tr.unsqueeze(P1, dim=1)
            P2 = tr.unsqueeze(P2, dim=1)

        return P1, P2


    def get_bdry(self):
        # TODO: Only for 2D
        assert self.d == 2
        # data structures
        P1 = tr.zeros(2*(self.N[0]+self.N[1]-2), 2, \
                      dtype=self.dtype, \
                      device=self.device)

        P2 = tr.zeros(2*(self.N[0]+self.N[1]-2), 2, \
                      dtype=self.dtype, \
                      device=self.device)


        # get the element defining knots in a specific order
        # order: bottom, top, left, right
        # P1
        P1[               :1*(self.N[0]-1),:] = \
                 tr.cartesian_prod(self.co[0][:-1],\
                                   tr.tensor([self.co[1][  0]],\
                                             dtype=self.dtype, \
                                             device=self.device))
        P1[1*(self.N[0]-1):2*(self.N[0]-1),:] = \
                 tr.cartesian_prod(self.co[0][:-1],\
                                   tr.tensor([self.co[1][ -1]],\
                                             dtype=self.dtype, \
                                             device=self.device))
        P1[2*(self.N[0]-1):2*(self.N[0]-1)+(self.N[1]-1),:] = \
                 tr.cartesian_prod(tr.tensor([self.co[0][  0]], \
                                             dtype=self.dtype, \
                                             device=self.device),\
                                   self.co[1][:-1])
        P1[2*(self.N[0]-1)+(self.N[1]-1):               ,:] = \
                 tr.cartesian_prod(tr.tensor([self.co[0][ -1]], \
                                             dtype=self.dtype, \
                                             device=self.device),\
                                   self.co[1][:-1])
        # P2
        P2[               :1*(self.N[0]-1),:] = \
                 tr.cartesian_prod(self.co[0][1: ],\
                                   tr.tensor([self.co[1][  0]], \
                                   dtype=self.dtype, \
                                   device=self.device))
        P2[1*(self.N[0]-1):2*(self.N[0]-1),:] = \
                 tr.cartesian_prod(self.co[0][1: ],\
                                   tr.tensor([self.co[1][ -1]], \
                                   dtype=self.dtype, \
                                   device=self.device))
        P2[2*(self.N[0]-1):2*(self.N[0]-1)+(self.N[1]-1),:] = \
                 tr.cartesian_prod(tr.tensor([self.co[0][  0]], \
                                   dtype=self.dtype, \
                                   device=self.device),\
                                   self.co[1][1: ])
        P2[2*(self.N[0]-1)+(self.N[1]-1):               ,:] = \
                 tr.cartesian_prod(tr.tensor([self.co[0][ -1]], \
                                             dtype=self.dtype, \
                                             device=self.device),\
                                   self.co[1][1: ])
        
        return P1, P2

    def outward_normal(self, P1, P2):
        # Output: outward normal vectors, shape=(numbdry,1,2)
        # Only for 2D
        assert self.d == 2
 
        # return the outward normal of boundary elements
        # the vector P2-P1/length rotated by pi/2 
        # is the outward normal vector, but the direction depends on the 
        # storage scheme
        normal = tr.sub(P2,P1)
        normal = tr.div(normal,tr.norm(normal,dim=1).unsqueeze(1))
        normal[1*(self.N[0]-1):2*(self.N[0]-1),:].mul_(-1)
        normal[2*(self.N[0]-1):2*(self.N[0]-1)+(self.N[1]-1),:].mul_(-1) 
        # switch coordinates which correspond to the rotation
        a = tr.empty(normal.shape[0], dtype=self.dtype, device=self.device)
        a[:] = normal[:,0]
        normal[:,0] = normal[:,1]
        normal[:,1] = -a

        return normal.unsqueeze(dim=1)


class embedded_unit_square():
    # unit square embedded in a bigger square
    def __init__(self, \
                 unit_square_Bspline_class, \
                 polygrad):

        # contains several methods and data structures, which has to be extended
        self.domain_disc = unit_square_Bspline_class
        self.polygrad    = polygrad
        # unit square dimension
        self.d      = self.domain_disc.d
        # number of knots in the embedded and embedding domain
        # number of knots which are added on both sides
        add_knots   = (polygrad)
        self.N_emb  = [self.domain_disc.N[i]+2*add_knots for i in range(self.d)]
        self.N      = self.domain_disc.N
 
        # for type and device agnostic code
        self.dtype  = self.domain_disc.dtype
        self.device = self.domain_disc.device

        # extend self.domain_disc
        self.co     = self.domain_disc.co
        self.co_emb = []
        for co in self.domain_disc.co:
            h = co[1]-co[0]
            points_before = tr.tensor([0-i*h for i in range(add_knots,0,-1)],
                                      dtype=self.dtype,
                                      device=self.device)

            points_after  = tr.tensor([1+i*h for i in range(1, add_knots+1)],
                                      dtype=self.dtype,
                                      device=self.device)
            self.co_emb.append(tr.cat((points_before, co, points_after), dim=0))


    def get_el(self):
        # get the element defining knots in a specific order
        P1 = tr.cartesian_prod(*[self.co_emb[i][ :-1] for i in range(self.d)])
        P2 = tr.cartesian_prod(*[self.co_emb[i][1:  ] for i in range(self.d)])
        return P1, P2


    # the boundary stays the same
    def get_initial(self):
        return self.domain_disc.get_initial()


    def get_bdry(self):
        return self.domain_disc.get_bdry()


    def outward_normal(self, P1, P2):
        return self.domain_disc.outward_normal(P1, P2)



### Discretization class, which discretizes a box around Omega (omega is a
### tensor product domain) such that it is convenient for wavelets
### Wavelets are periodized.
class wavelet_box_discretization_per():
    def __init__(self,
                 wname,
                 level,
                 omega=[(0,1)],
                 omega_eq_box=False,
                 box=None,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        # Input:
        #   -wname: the wavelet name code, e.g. 'bior3.7'
        #   -level: the level used for discretization
        #   -omega: list of intervals that span the domain, len(omega) = dim
        #           (space or space+time dimension)
        #   -omega_eq_box: bool if omega \equiv box
        #   -box         : list of intervals that span box, len(box) = dim
        #           (space or space+time dimension), if box is not None it will
        #           be checked if omega is equal to grid points

        # REMARK: The typical "coordinates", "elements" data structures are not
        #         needed in this box discretization, because the scaling
        #         functions define the discretization. With their tensor
        #         structure they define axis parallel rectangles which can be
        #         described with two diagonal verteces each.

        # data type and device
        self.dtype  = dtype
        self.device = device

        # wavelet type
        _, self.d, self.d_tilde = wname_to_code(wname)
        # maximal level
        self.J = level

        # unit square dimension
        self.dim    = len(omega)
        # bool if omega \equiv box
        self.omega_eq_box = omega_eq_box
        # the domain omega
        self.omega  = omega
        # box domain
        self.box    = [()]*self.dim


        # if omega is box
        if self.omega_eq_box is True:
            # box domain
            self.box   = self.omega

            # number of nodes in \Box
            self.N_box = 2**self.J+1

            # number of nodes in \Omega
            self.N     = self.N_box

            # Define number of zero entries in c_J on the left and right side
            nz = 2**self.J + 1 - (self.d+1)//2

        elif self.omega_eq_box is False:
            # To determine the box, we require that each (a,b) are support 
            # boundaries for some scaling functions and only
            #   - supp phi_{j,0}^{per} \cap (a,b)      = \emptyset
            #   - supp phi_{j,2**j-1}^{per} \cap (a,b) = \emptyset
            # => c_j is periodic

            # Define number of zero entries in c_J on the left and right side
            nz = 10

            # Matrix for the lin. sys. of eq. to enforce the conditions above.
            c1 = ceil(self.d/2) + (nz-1)
            c2 = ceil(self.d/2) + (self.d+1)%2  + (nz-1)

            A = tr.tensor([[1-2**(-self.J)*c1, \
                               2**(-self.J)*c1], \
                           [2**(-self.J)*c2, \
                               1-2**(-self.J)*c2]], \
                           dtype=self.dtype)

            # calculate for each interval the extended interval
            for i, interval in enumerate(self.omega):
                a = interval[0]
                b = interval[1]

                # the rhs for the lin. sys. of eq.
                r = tr.tensor([[a],[b]], dtype=dtype)
                # check if the level is too small
                if tr.abs(tr.linalg.det(A)) < 1e-8:
                    raise Exception(\
                            'Box domain error: - increase level j\n'+\
                            '                  - decrease wavelet order d')

                # solve the lin. sys. of eq.
                sol = tr.linalg.solve(A,r)
                a_tilde = sol[0,0]
                b_tilde = sol[1,0]
                # check if the level is too small
                if a_tilde >= b_tilde:
                    raise Exception(\
                            'Box domain error: - increase level j\n'+\
                            '                  - decrease wavelet order d')

                self.box[i] = (a_tilde,b_tilde)


            # Now discretize \Box, extract the discretization for \Omega and
            # throw away the remaining parts

            # number of nodes in \Box
            self.N_box = 2**self.J+1

            # number of nodes in \Omega (is the same in each dimension, given by
            # construction of \Box)
            self.N     = self.N_box-(self.d+1) - 2*(nz-1)

        elif box is not None:
            ### discretize box first
            # box
            self.box = box

            # number of nodes in \Box
            self.N_box = 2**self.J+1

            # check if omega is equal to grid points
            for dim, (a,b) in enumerate(self.omega):
                check_flag = 0
                a_tilde = self.box[dim][0]
                b_tilde = self.box[dim][1]

                # check a
                val = (a-a_tilde)/(b_tilde-a_tilde) * 2**self.J
                check_flag = floor(val)==ceil(val)
                if not check_flag:
                    raise ValueError('If \Box is given \Omega must be equal '+\
                                     'to grid points')
                # check b
                val = (b_tilde-b)/(b_tilde-a_tilde) * 2**self.J
                check_flag = floor(val)==ceil(val)
                if not check_flag:
                    raise ValueError('If \Box is given \Omega must be equal '+\
                                     'to grid points')

            ### if omega is equal to grid points
            # number of nodes in \Omega
            self.N = self.N_box - floor((a-a_tilde)/(b_tilde-a_tilde)*2**J) \
                                - floor((b_tilde-b)/(b_tilde-a_tilde)*2**J)
 
        else:
            raise ValueError('\Omega must be given.')

        # Number of elements in \Omega
        self.nelem = (self.N-1)**self.dim
        # Number of elements in \Omega in each dim
        self.nelem_dim  = self.N-1

        # Discretization for \Omega. The structure "coordinates" can be received
        # with a tensor product of all entries in the list. We do not need to
        # store it explicitly.
        self.co = [tr.linspace(a, b, self.N, \
                               dtype=self.dtype, \
                               device=self.device) \
                                    for (a,b) in self.omega]


        # Get the element defining coordinates in a specific order (the torch
        # function cartesian_prod defines the order)
        # shape = (2, nelem, dim)
        self.elem_coords = tr.zeros(2, self.nelem, self.dim, \
                                    dtype=self.dtype,
                                    device=self.device)
        self.elem_coords[0,:,:] = tr.cartesian_prod(*[self.co[i][ :-1] \
                                                    for i in range(self.dim)])\
                                                  .reshape(self.nelem, self.dim)
        self.elem_coords[1,:,:] = tr.cartesian_prod(*[self.co[i][1:  ] \
                                                    for i in range(self.dim)])\
                                                  .reshape(self.nelem, self.dim)


        # Get the interval to function information, which means the information
        # for which functions the intersection
        # supp(phi_j,k) \cap [x_i, x_i+1] != \emptyset.
        # In the particular case of scaling functions this information is the
        # "position in space" k. Due to the tensor structure of the scaling
        # functions it is the same for an interval in each dimension. A tensor
        # product of this gives the global indexing.

        # shape = (self.N-1, d)
        self.interval2func = tr.zeros(self.N-1, self.d, \
                                      dtype=tr.long, \
                                      device=self.device)

        # By construction of \Box it starts with k=nz and ends with k=2**j-nz-1.
        # Also by construction the first self.d-1 entries in the first and last
        # row intersect the boundary.
        for i in range(self.interval2func.shape[0]):
            self.interval2func[i,:] = tr.arange(nz+i, nz+i+self.d, 1, \
                                                dtype=tr.long, \
                                                device=self.device)

        # if omega \equiv box, then the periodized functions near the boundary
        # need special consideration
        self.interval2func = tr.remainder(self.interval2func,2**self.J)


    def get_initial(self):
        # Get the initial (=means time t=0, first dimension is assumed to be the
        # time) element defining coordinates in a specific order (the torch
        # function cartesian_prod defines the order). Mostly for initial
        # conditions.
        initial_size = (self.N-1)**(self.dim-1)
        # shape = (2, nelem, dim)
        initial = tr.zeros(2, initial_size, self.dim, \
                                dtype=self.dtype,
                                device=self.device)
        initial[0,:,:] = tr.cartesian_prod(tr.zeros(1, \
                                        dtype=self.dtype, \
                                        device=self.device), \
                               *[self.co[i][ :-1] for i in range(1,self.dim)])

        initial[1,:,:] = tr.cartesian_prod(tr.zeros(1, \
                                        dtype=self.dtype, \
                                        device=self.device), \
                               *[self.co[i][1:  ] for i in range(1,self.dim)])


        return initial


    def get_bdry(self):
        # boundary coordinates
        # 2 points descripe the boundary elements, per dimension, 2 boundaries,
        # of nbdry bdry elements in \mathbb{R}^{dim}
        # shape = (2, dim, 2, nbdry, dim)
        bdry_coords = tr.zeros(2, \
                               self.dim, \
                               2, \
                               (self.N-1)**(self.dim-1), \
                               self.dim, \
                               dtype=self.dtype, \
                               device=self.device)

        bdry_list_P1 = [self.co[i][ :-1] for i in range(self.dim)]
        bdry_list_P2 = [self.co[i][1:  ] for i in range(self.dim)]

        for _dim in range(self.dim):
            # P1 / P2
            #   dim
            #       first / second bdry
            #           bdry elements
            #               \mathbb{R}^{dim}

            ## P1
            bdry_list_P1_tmp          = copy.copy(bdry_list_P1)
            # first border in that dimension
            bdry_list_P1_tmp[_dim]    = self.co[_dim][0].reshape(1)
            bdry_coords[0,_dim,0,:,:] = tr.cartesian_prod(*bdry_list_P1_tmp)

            # second border in that dimension
            bdry_list_P1_tmp[_dim]    = self.co[_dim][-1].reshape(1)
            bdry_coords[0,_dim,1,:,:] = tr.cartesian_prod(*bdry_list_P1_tmp)

            ## P2
            bdry_list_P2_tmp          = copy.copy(bdry_list_P2)
            # first border in that dimension
            bdry_list_P2_tmp[_dim]    = self.co[_dim][0].reshape(1)
            bdry_coords[1,_dim,0,:,:] = tr.cartesian_prod(*bdry_list_P2_tmp)

            # second border in that dimension
            bdry_list_P2_tmp[_dim]    = self.co[_dim][-1].reshape(1)
            bdry_coords[1,_dim,1,:,:] = tr.cartesian_prod(*bdry_list_P2_tmp)


        # reshape to the usual shape=(2, nbdry, dim)
        bdry_coords = bdry_coords.reshape(2,\
                                          self.dim*2*(self.N-1)**(self.dim-1),\
                                          self.dim)

        return bdry_coords


    def outward_normal(self, P1, P2):
        # TODO: in this case the normal vectors are the unit vectors!
        # Output: outward normal vectors, shape=(numbdry,1,2)
        # Only for 2D
        assert self.dim == 2

        # return the outward normal of boundary elements
        # the vector P2-P1/length rotated by pi/2
        # is the outward normal vector, but the direction depends on the
        # storage scheme
        normal = tr.sub(P2,P1)
        normal = tr.div(normal,tr.norm(normal,dim=1).unsqueeze(1))
        normal[1*(self.N[0]-1):2*(self.N[0]-1),:].mul_(-1)
        normal[2*(self.N[0]-1):2*(self.N[0]-1)+(self.N[1]-1),:].mul_(-1)
        # switch coordinates which correspond to the rotation
        a = tr.empty(normal.shape[0], dtype=self.dtype, device=self.device)
        a[:] = normal[:,0]
        normal[:,0] = normal[:,1]
        normal[:,1] = -a

        return normal.unsqueeze(dim=1)



### Discretization class, which discretizes a box around \Omega (omega is a
### tensor product domain) such that it is convenient for wavelets.
### The wavelets are Not periodized.
class wavelet_box_discretization():
    def __init__(self, 
                 wname,
                 level,
                 omega=[(0,1)], 
                 dtype=tr.double,
                 device=tr.device('cpu')):

        # Input: 
        #   -wname: the wavelet name code, e.g. 'bior3.7'
        #   -level: the level used for discretization
        #   -omage: list of intervals that span the domain, len(omega) = dim 
        #           (space or space+time dimension)

        # REMARK: The typical "coordinates", "elements" data structures are not 
        #         needed in this box discretization, because the scaling 
        #         functions define the discretization. With their tensor 
        #         structure they define axis parallel rectangles which can be
        #         described with two diagonal verteces each.

        # data type and device
        self.dtype  = dtype
        self.device = device

        # wavelet type
        _, self.d, self.d_tilde = wname_to_code(wname)
        # maximal level
        self.J = level

        # unit square dimension
        self.dim    = len(omega)
        # the domain omega
        self.omega  = omega

        ############################# Real line case ###########################
        # check if Omega is defined through dyadic grid points
        j = 0
        for val in self.omega:
            ja, jb = get_level(val[0]), get_level(val[1])
            if ja == -1 or jb == -1:
                raise Exception("\Omega is not defined on dyadic grid points")
            j = max(ja, jb, j)

        # check if the finest level is fine enough
        if self.J < j:
            raise Exception("The level is too small")

        # number of nodes in \Omega
        self.N          = [int((val[1]-val[0])*2**self.J + 1) \
                                for val in self.omega]
        # Number of elements in \Omega in each dim
        self.nelem_dim  = [n-1 for n in self.N]
        # Number of elements in \Omega
        self.nelem      = math.prod(self.nelem_dim)
        # Number of scaling functions overlapping the domain \Omega in each dim
        self.nfunc_dim  = [2**self.J*(b-a)+self.d-1 for (a,b) in self.omega]
        # Number of scaling functions overlapping the domain \Omega
        self.nfunc      = math.prod(self.nfunc_dim)
        # Number of bdry elements in \Omega
        self.nbdry = 0
        for _dim in range(self.dim):
            self.nbdry += 2*math.prod(self.nelem_dim[:_dim])\
                           *math.prod(self.nelem_dim[_dim+1:])


        # Discretization for \Omega. The structure "coordinates" can be received
        # with a tensor product of all entries in the list. We do not need to 
        # store it explicitly.
        self.co = [tr.linspace(self.omega[_dim][0], \
                               self.omega[_dim][1], 
                               self.N[_dim], \
                               dtype=self.dtype, \
                               device=self.device) \
                                for _dim in range(self.dim)]

        # Axis parallel quadrilaterals. 
        # Get the element defining coordinates in a specific order (the torch 
        # function cartesian_prod defines the order)
        # shape = (2, nelem, dim)
        self.elem_coords = tr.zeros(2, self.nelem, self.dim, \
                                    dtype=self.dtype, 
                                    device=self.device)
        self.elem_coords[0,:,:] = tr.cartesian_prod(*[self.co[i][ :-1] \
                                                    for i in range(self.dim)])\
                                                  .reshape(self.nelem, self.dim)
        self.elem_coords[1,:,:] = tr.cartesian_prod(*[self.co[i][1:  ] \
                                                    for i in range(self.dim)])\
                                                  .reshape(self.nelem, self.dim)


        # Get the interval to function information, which means the information
        # for which functions the intersection 
        # supp(phi_j,k) \cap [x_i, x_i+1] != \emptyset.
        # In the particular case of scaling functions this information is the 
        # "position in space" k. 


        # shape = [(self.N-1, d)]*dim
        self.interval2func = [tr.zeros(self.N[_dim]-1, self.d, \
                                      dtype=tr.long, \
                                      device=self.device) \
                              for _dim in range(self.dim)]

        for _dim in range(self.dim):
            for i in range(self.interval2func[_dim].shape[0]):
                a = self.omega[_dim][0]
                self.interval2func[_dim][i,:] =  \
                         tr.arange(int(2**self.J*a)-ceil(self.d/2)+i+1, \
                                   int(2**self.J*a)-ceil(self.d/2)+i+self.d+1, \
                                   1, \
                                   dtype=tr.long, \
                                   device=self.device)

        ########################## End of real line case #######################

    def get_initial(self):
        # Get the initial (=means time t=0, first dimension is assumed to be the
        # time) element defining coordinates in a specific order (the torch 
        # function cartesian_prod defines the order). Mostly for initial 
        # conditions.
        initial_size = (self.N-1)**(self.dim-1)
        # shape = (2, nelem, dim)
        initial = tr.zeros(2, initial_size, self.dim, \
                                dtype=self.dtype, 
                                device=self.device)
        initial[0,:,:] = tr.cartesian_prod(tr.zeros(1, \
                                        dtype=self.dtype, \
                                        device=self.device), \
                               *[self.co[i][ :-1] for i in range(1,self.dim)])

        initial[1,:,:] = tr.cartesian_prod(tr.zeros(1, \
                                        dtype=self.dtype, \
                                        device=self.device), \
                               *[self.co[i][1:  ] for i in range(1,self.dim)])


        return initial


    def get_bdry(self):
        # Boundary coordinates and the respective normal vectors (these are the
        # unit vectors)
        # 2 points descripe the boundary elements, per dimension, 2 boundaries,
        # of nbdry bdry elements in \mathbb{R}^{dim}
        # shape = (2, dim, 2, nbdry of one side, dim)
        bdry_coords = tr.zeros(2, \
                               self.dim, \
                               2, \
                               (self.N-1)**(self.dim-1), \
                               self.dim, \
                               dtype=self.dtype, \
                               device=self.device)

        bdry_list_P1 = [self.co[i][ :-1] for i in range(self.dim)]
        bdry_list_P2 = [self.co[i][1:  ] for i in range(self.dim)]

        # outward normal vectors
        I = tr.eye(self.dim, dtype=self.dtype, device=self.device)
        normal = tr.zeros(self.dim,\
                          2, \
                          (self.N-1)**(self.dim-1), \
                          self.dim, \
                          dtype=self.dtype, \
                          device=self.device)



        for _dim in range(self.dim):
            # P1 / P2
            #   dim
            #       first / second bdry
            #           bdry elements
            #               \mathbb{R}^{dim}

            ## P1
            bdry_list_P1_tmp = copy.copy(bdry_list_P1)
            # first border in that dimension
            bdry_list_P1_tmp[_dim] = self.co[_dim][0].reshape(1)
            bdry_coords[0,_dim,0,:,:] = tr.cartesian_prod(*bdry_list_P1_tmp)

            # second border in that dimension
            bdry_list_P1_tmp[_dim] = self.co[_dim][-1].reshape(1)
            bdry_coords[0,_dim,1,:,:] = tr.cartesian_prod(*bdry_list_P1_tmp)

            ## P2
            bdry_list_P2_tmp = copy.copy(bdry_list_P2)
            # first border in that dimension
            bdry_list_P2_tmp[_dim] = self.co[_dim][0].reshape(1)
            bdry_coords[1,_dim,0,:,:] = tr.cartesian_prod(*bdry_list_P2_tmp)

            # second border in that dimension
            bdry_list_P2_tmp[_dim] = self.co[_dim][-1].reshape(1)
            bdry_coords[1,_dim,1,:,:] = tr.cartesian_prod(*bdry_list_P2_tmp)

            # normal vectors
            normal[_dim, 0, :, :] = -I[_dim,:]
            normal[_dim, 1, :, :] =  I[_dim,:]


        # reshape to the usual shape=(2, nbdry, dim)
        bdry_coords = bdry_coords.reshape(2,\
                                          self.nbdry,\
                                          self.dim)

        ## normal vectors
        normal = normal.reshape(self.nbdry, \
                                self.dim)
 

        return bdry_coords, normal




# Discretization of box with tensor elements on which the per. primal B-spline
# wavelets are defined
class per_wavelet_box_disc():
    def __init__(self,
                 wname,
                 level,
                 box=[(0,1)],
                 dtype=tr.double,
                 device=tr.device('cpu')):

        # Input:
        #   -wname: the wavelet name code, e.g. 'bior3.7'
        #   -level: the level used for discretization
        #   -box  : list of intervals that span box, len(box) = dim

        # REMARK: The typical "coordinates", "elements" data structures are not
        #         needed in this box discretization, because the scaling
        #         functions define the discretization. With their tensor
        #         structure they define axis parallel rectangles which can be
        #         described with two diagonal verteces each.

        # data type and device
        self.dtype  = dtype
        self.device = device

        # wavelet type
        _, self.d, self.d_tilde = wname_to_code(wname)
        # maximal level
        self.J = level

        # unit square dimension
        self.dim    = len(box)
        # the domain box 
        self.box    = box 

        # number of nodes in \Box
        self.N = 2**self.J+1

        # Define number of zero entries in c_J on the left and right side
        nz = 2**self.J + 1 - (self.d+1)//2

        # Number of elements in \Omega
        self.nelem = (self.N-1)**self.dim
        # Number of elements in \Omega in each dim
        self.nelem_dim  = self.N-1

        # Discretization for \Omega. The structure "coordinates" can be received
        # with a tensor product of all entries in the list. We do not need to
        # store it explicitly.
        self.co = [tr.linspace(a, b, self.N, \
                               dtype=self.dtype, \
                               device=self.device) \
                                    for (a,b) in self.box]


        # Get the element defining coordinates in a specific order (the torch
        # function cartesian_prod defines the order)
        # shape = (2, nelem, dim)
        self.elem_coords = tr.zeros(2, self.nelem, self.dim, \
                                    dtype=self.dtype,
                                    device=self.device)
        self.elem_coords[0,:,:] = tr.cartesian_prod(*[self.co[i][ :-1] \
                                                    for i in range(self.dim)])\
                                                  .reshape(self.nelem, self.dim)
        self.elem_coords[1,:,:] = tr.cartesian_prod(*[self.co[i][1:  ] \
                                                    for i in range(self.dim)])\
                                                  .reshape(self.nelem, self.dim)


        # Get the interval to function information, which means the information
        # for which functions the intersection
        # supp(phi_j,k) \cap [x_i, x_i+1] != \emptyset.
        # In the particular case of scaling functions this information is the
        # "position in space" k. Due to the tensor structure of the scaling
        # functions it is the same for an interval in each dimension. A tensor
        # product of this gives the global indexing.

        # shape = (self.N-1, d)
        self.interval2func = tr.zeros(self.N-1, self.d, \
                                      dtype=tr.long, \
                                      device=self.device)

        # By construction of \Box it starts with k=nz and ends with k=2**j-nz-1.
        # Also by construction the first self.d-1 entries in the first and last
        # row intersect the boundary.
        for i in range(self.interval2func.shape[0]):
            self.interval2func[i,:] = tr.arange(nz+i, nz+i+self.d, 1, \
                                                dtype=tr.long, \
                                                device=self.device)

        # if omega \equiv box, then the periodized functions near the boundary
        # need special consideration
        self.interval2func = tr.remainder(self.interval2func,2**self.J)


