import torch as tr
from torch.nn.modules.module import Module



############################## Neural Networks #################################


class Dense_NN(Module):
    # Class for a dense NN with variable size.
    # Input:
    #    -nn_list = [input dim., first hidden layer size,...,
    #                last hidden layer size, output dim.]

    def __init__(self, \
                 nn_list, \
                 activation_func=tr.nn.functional.relu, \
                 dtype=tr.double, \
                 device=tr.device('cpu')):

        super().__init__()
        self.dtype      = dtype
        self.device     = device
        self.nn_list    = nn_list
        self.activation = activation_func
        self.hidden     = tr.nn.ModuleList()
        for i in range(len(nn_list)-1):
            self.hidden.append(tr.nn.Linear(nn_list[i], \
                                            nn_list[i+1], \
                                            dtype=self.dtype, \
                                            device=self.device))

    def forward(self, x):
        # forward pass through the network
        for layer in self.hidden[:-1]:
            x = self.activation(layer(x))
        # the last layer is a linear layer without activation function
        output = self.hidden[-1](x)

        return output



########################## NNs with exact bdry condtions #######################

class NN_dirichlet(Dense_NN):
    # NN-class for a dense NN with variable size, with zero bdry condition
    # Input:
    #    -nn_list = [input dim., first hidden layer size,...,
    #                last hidden layer size, output dim.]

    def __init__(self, \
                 nn_list, \
                 distance_func, \
                 activation_func=tr.nn.functional.relu, \
                 dtype=tr.double, \
                 device=tr.device('cpu')):

        super().__init__(nn_list, \
                         activation_func=activation_func, \
                         dtype=dtype, \
                         device=device)

        # assign the distance function which models the bdry conditions
        self.dist_func = distance_func


    def forward(self, x):
        # eval the approximate distance function
        dist_func_values = self.dist_func(x)

        # forward pass through the network
        for layer in self.hidden[:-1]:
            x = self.activation(layer(x))
        # the last layer is a linear layer without activation function
        output = dist_func_values*self.hidden[-1](x)

        return output

 
class NN_dirichlet_param(Dense_NN):
    # NN-class for a dense NN with variable size, with zero bdry condition
    # Input:
    #    -nn_list = [input dim., first hidden layer size,...,
    #                last hidden layer size, output dim.]

    def __init__(self, \
                 nn_list, \
                 distance_func, \
                 activation_func=tr.nn.functional.relu, \
                 dtype=tr.double, \
                 device=tr.device('cpu')):

        super().__init__(nn_list, \
                         activation_func=activation_func, \
                         dtype=dtype, \
                         device=device)

        # assign the distance function which models the bdry conditions
        self.dist_func = distance_func


    def forward(self, x):
        # eval the approximate distance function
        dist_func_values = self.dist_func(x, mu=x.reshape(-1,x.shape[-1])[0,-1])

        # forward pass through the network
        for layer in self.hidden[:-1]:
            x = self.activation(layer(x))
        # the last layer is a linear layer without activation function
        output = dist_func_values*self.hidden[-1](x)

        return output





