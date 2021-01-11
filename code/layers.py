import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class   GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self,in_size,out_size,bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = in_size
        self.out_features = out_size
        '''参数定义'''
        self.weight = Parameter(torch.FloatTensor(in_size, out_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_size))
        else:
            self.register_parameter('bias', None)
        '''初始化参数'''
        self.reset_parameters()

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class   Highway_dense(Module):
    '''
     y = t * h1 + (1 - t) * h2
    '''
    def __init__(self,in_size,out_size):
        super(Highway_dense,self).__init__()

        self.in_features = in_size
        self.out_features = out_size
        self.gconv = GraphConvolution(in_size, out_size)
        self.linear = nn.Linear(in_size, out_size,bias=True)
        '''初始化参数'''
        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant(self.linear.bias, -4.0)

    def forward(self, x, adj):
        #graphconv layer
        l_h = self.gconv(x,adj)
        l_h = F.tanh(l_h)

        # gate layer
        l_t = self.linear(x)
        l_t = F.sigmoid(l_t)

        #   y = t * h1 + (1 - t) * h2
        output=l_t * l_h + (1.0 - l_t) * x

        return output







