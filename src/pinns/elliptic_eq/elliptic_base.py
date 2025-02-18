


###### PINN
# Variational Network for poissons equation with periodic bdry conditions and 
# with wavelet extension in nD (Base Class)
class VarNet_elliptic_wavelet_per_nD_base():
    def __init__(self,
                 nn_model,
                 box_disc,
                 testcase,
                 F_K,
                 detJF_K,
                 alpha,
                 support_vec,
                 mu,
                 wavelet,
                 dtype,
                 device):

        # data type
        self.dtype = dtype
        # device
        self.device = device

        # sobolev regularity 
        self.sigma = None

        # neural network
        self.nn_model = nn_model

        # mus for PPDEs
        self.mu = mu

        # discretization
        self.box_disc       = box_disc
        # dimension
        self.dim            = self.box_disc.dim
        # coordinate vectors in each dimension (tensor discretization)
        self.co             = self.box_disc.co
        # number of coordinates in each dimension
        self.N              = self.box_disc.N
        # number of elements
        self.nelem          = self.box_disc.nelem
        # number of elements
        self.nelem_dim      = self.box_disc.nelem_dim
        # element coordinates
        self.elem_coords    = self.box_disc.elem_coords
        # interval to scaling function indeces
        self.interval2func  = self.box_disc.interval2func
         # domain omega
        self.omega          = self.box_disc.omega
        # domain box
        self.box            = self.box_disc.box
 

        # wavelet transformation
        self.wavelet = wavelet

        # FWT class
        self.fwt    = periodic_fwt_nD(wavelet=wavelet, \
                                      decomp_mode='primal', \
                                      dtype=self.dtype, \
                                      device=self.device)

        # wavelet function class (scaling functions are also available)
        self.wfunc = self.fwt.w
        # order of primal scaling/wavelet functions
        self.d     = self.wfunc.d
        # maximal level
        self.level = self.box_disc.J

        # testcase
        self.testcase    = testcase

        # affine transformation on reference element [-1,1]
        self.F_K         = F_K   
        self.detJF_K     = detJF_K   

        ## quadrature 1D: sum_{i=1,...,n} alpha_i*f(support_vec_i)
        # number of support vectors (degree of the quadrature rule)
        self.n           = support_vec.shape[0]
        # weights
        self.alpha       = alpha.to(dtype=dtype, device=device)
        # support vectors
        self.support_vec = support_vec.to(dtype=dtype, device=device)
        # quadrature nD
        self.X_hat = tr.cartesian_prod(*[self.support_vec]*self.dim) \
                                       .to(dtype=dtype, device=device)

        # torch cartesian product return wrong shape of X_hat in 1D
        if self.dim == 1:
            self.X_hat = self.X_hat.unsqueeze(1)

        # the associated weights
        self.Alpha = tr.unsqueeze(kron_nD([self.alpha]*self.dim),dim=0)\
                                  .to(dtype=dtype, device=device)

        # space
        self.Phi_1D  = tr.zeros(1, \
                                self.d, \
                                self.n, \
                                dtype=dtype, \
                                device=device)
 
        # fill the arrays
        # for all elements
        for i in range(1):
            X = self.F_K(self.support_vec, \
                         self.co[0][i].reshape(1,1), \
                         self.co[0][i+1].reshape(1,1)).reshape(-1)

            # for all nonzero basis functions
            # Remark: The scaling functions are periodized upon \Box not 
            #         \Omega
            for k in range(self.d):
                self.Phi_1D[i,k,:] = self.wfunc.per_primal_scal(\
                                           X, \
                                           self.level, \
                                           self.interval2func[i,k], \
                                           interval=self.box_disc.box[0])


        # Phi
        Phi_list    = [self.Phi_1D]*self.dim
        self.Phi_nD = kron_nD(Phi_list)

        ## non-parameter-dependent data structures 
        # (frequently used during training)
        # all elements T = [x_i, x_i+1]**dim
        # shape = (numel,dim)
        P1 = self.elem_coords[0,:,:]
        P2 = self.elem_coords[1,:,:]

        # detJF_K
        # shape = (numel, 1, 1)
        self.absdetJF_K = tr.abs(self.detJF_K(P1,P2))

        # transform support vectors
        # shape = (numel, n**dim, dim)
        self.X = self.F_K(self.X_hat, P1, P2)

        # auxilary data structures (frequently used during training)
        self.aux_ones = tr.ones(self.nelem, \
                                self.n**self.dim, \
                                mu[0].shape[0], \
                                device=device, \
                                dtype=dtype)

        ### Indeces for nD case
        tensor_ones = tr.ones(self.interval2func.shape,\
                              dtype=tr.long,\
                              device=self.device)

        ones_list     = [tensor_ones]*self.dim
        ones_list[-1] = self.interval2func
        self.global_i = kron_nD(ones_list)
        ones_list[-1] = tensor_ones
        for dim_ in range(1, self.dim):
            ones_list[-dim_-1] = self.interval2func
            self.global_i =   self.global_i \
                            + (2**(dim_*self.level))*kron_nD(ones_list)
            ones_list[-dim_-1] = tensor_ones


        # shape: (1,2**(level*dim)*d**dim)
        self.global_i = self.global_i.reshape(-1).unsqueeze(0)


    def evaluate(self, **kwargs):
        loss = 0.
        for i in range(self.mu.shape[0]):
            # get the mu
            mu   = self.mu[i,:]

            loss = loss + self.evaluate_mu(mu)

        return loss


    def evaluate_mu(self, mu):
        # Is overwritten in children classes
        raise ValueError('The base class can not be evaluated for one \mu')

        return None


    def get_wave_coeffs(self, mu):
        # get the signal for one mu
        signal = self.get_signal(mu)

        # transform the signal with the FWT
        AD = self.fwt.multilevel_dwt(signal)

        return AD


    def get_signal(self, mu):
        # Is overwritten in children classes
        raise ValueError('The base class can not be evaluated')

        return None




