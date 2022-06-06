import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import pdb
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out


#1*1conv
class Conv1x1(nn.Module):
    #1x1 convolution + bn + relu.
    
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0,
                              bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x 
# dynamic conditional conv layer (it is fc layer when the kernel size is 1x1 and the input is cx1x1)
class ConvBasis2d(nn.Module):
    def __init__(self, idfcn, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, transposed=False, output_padding=_pair(0), groups=1, bias=True):
        super(ConvBasis2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.idfcn = idfcn  # the dimension of coditional input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.weight_basis = Parameter(torch.Tensor(idfcn*out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight_basis.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input, idw):
        #pdb.set_trace()
        # idw: conditional input
        output = F.conv2d(input, self.weight_basis, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = output.view(output.size(0), self.idfcn, self.out_channels, output.size(2), output.size(3)) * \
                 idw.view(-1, self.idfcn,1, 1, 1).expand(output.size(0), self.idfcn, self.out_channels, output.size(2), output.size(3))
        output = output.sum(1).view(output.size(0), output.size(2), output.size(3), output.size(4))
        return output