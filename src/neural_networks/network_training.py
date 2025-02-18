import numpy as np
import torch as tr

class ScipyWrapper_PINN():
    def __init__(self,
                 params,
                 model,
                 lossfn,
                 dtype=tr.double,
                 device=tr.device('cpu')):

        self._params      = list(params)
        self._numel_cache = None
        self._model       = model
        self._lossfn      = lossfn
        self.nump         = len(self._params)
        self.numparams    = 0
        self.device_      = device

        if dtype is tr.double:
            self.trdtype = tr.double
            self.npdtype = np.float64
        else:
            self.trdtype = tr.float32
            self.npdtype = np.float32 

        for p in self._params:
            self.numparams += p.numel()


    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return tr.cat(views, 0)


    def _set_param(self, params_data):
        for i, dat in enumerate(params_data):
            self._params[i].data = dat.data

    def _flat_param(self):
        x      = np.empty(self.numparams, dtype=self.npdtype)
        offset = 0
        for p in self._params:
            numelp = p.numel()
            x[offset:offset+numelp] = p.reshape(numelp).cpu().detach()\
                                       .numpy().astype(dtype=self.npdtype)
            offset += numelp
        return x

    def _build_param(self, x):
        params = [False]*self.nump
        offset = 0
        for i, p in enumerate(self._params):
            numelp = p.numel()
            temp = tr.from_numpy(x[offset:offset+numelp]).reshape(p.shape)
            params[i] = temp.data.clone().to(dtype=self.trdtype, \
                                             device=self.device_).detach()
            params[i].requires_grad = True
            offset += numelp

        return params


    def _zero_grad(self, set_to_none: bool = False):
        for p in self._params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()
    

    def _eval_function(self, x_np, **kwargs):
        x_tr = self._build_param(x_np)
        self._set_param(x_tr)
        # there is no input to the model in this scipy_wrapper because for the
        # uwVarNet the model gets evaluated within the lossfn
        loss_tr = self._lossfn()
        loss_np = loss_tr.cpu().detach().numpy().astype(dtype=self.npdtype)
        return loss_np

    def _eval_gradient(self, x_np, **kwargs):
        x_tr = self._build_param(x_np)
        self._set_param(x_tr)
        self._zero_grad()
        loss_tr = self._lossfn()
        loss_tr.backward()
        flat_grad_tr = self._gather_flat_grad()
        flat_grad_np = flat_grad_tr.cpu().numpy().astype(dtype=self.npdtype)

        return flat_grad_np
        


########## scipy wrapper for device agnostic code 
class ScipyWrapper():
    def __init__(self,
                 params,
                 model,
                 trainset,
                 targets,
                 lossfn,
                 dtype=tr.double,
                 device=tr.device("cpu")):

        self.device_      = device

        if dtype is tr.double:
            self.trdtype = tr.double
            self.npdtype = np.float64
        else:
            self.trdtype = tr.float32
            self.npdtype = np.float32 


        self._params      = list(params)
        self._numel_cache = None
        self._model       = model
        self._trainset    = trainset.to(dtype=self.trdtype, device=self.device_)
        self._targets     = targets.to(dtype=self.trdtype, device=self.device_)
        self._lossfn      = lossfn
        self.nump         = len(self._params)
        self.numparams    = 0

        for p in self._params:
            self.numparams += p.numel()


    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return tr.cat(views, 0)


    def _set_param(self, params_data):
        for i, dat in enumerate(params_data):
            self._params[i].data = dat.data

    def _flat_param(self):
        x      = np.empty(self.numparams, dtype=self.npdtype)
        offset = 0
        for p in self._params:
            numelp = p.numel()
            x[offset:offset+numelp] = p.reshape(numelp).cpu().detach()\
                                       .numpy().astype(dtype=self.npdtype)
            offset += numelp
        return x

    def _build_param(self, x):
        params = [False]*self.nump
        offset = 0
        for i, p in enumerate(self._params):
            numelp = p.numel()
            temp = tr.from_numpy(x[offset:offset+numelp]).reshape(p.shape)
            params[i] = temp.data.clone().to(dtype=self.trdtype, \
                                             device=self.device_).detach()
            params[i].requires_grad = True
            offset += numelp

        return params

    def _zero_grad(self, set_to_none: bool = False):
        for p in self._params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()
    

    def _eval_function(self, x_np, **kwargs):
        x_tr = self._build_param(x_np)
        self._set_param(x_tr)
        output_model = self._model(self._trainset)
        loss_tr = self._lossfn(output_model, self._targets)
        loss_np = loss_tr.cpu().detach().numpy().astype(dtype=self.npdtype)
        return loss_np

    def _eval_gradient(self, x_np, **kwargs):
        x_tr = self._build_param(x_np)
        self._set_param(x_tr)
        self._zero_grad()
        output_model = self._model(self._trainset)
        loss_tr = self._lossfn(output_model, self._targets)
        loss_tr.backward()
        flat_grad_tr = self._gather_flat_grad()
        flat_grad_np = flat_grad_tr.cpu().numpy().astype(dtype=self.npdtype)

        return flat_grad_np
 
