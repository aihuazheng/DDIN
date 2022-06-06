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

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)] 
        feat_block += [nn.BatchNorm1d(low_dim)]
        
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x
        
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []       
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        x = self.classifier(x)
        return x  
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
#3*3
class Conv3x3(nn.Module):
    #1x1 convolution + bn + relu.
    
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=0,
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
# Define the ResNet18-based Model
class visible_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18',gm_pool = 'on'):
        super(visible_net_resnet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)

        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.visible = model_ft
        self.dropout = nn.Dropout(p=0.5)
        self.gm_pool = gm_pool
       
    def forward(self, x):
        #pdb.set_trace()
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
       # x_s1 = x
       # x_s1_avg = self.visible.avgpool(x_s1)
       # x_s1_f = x_s1_avg.view(x_s1_avg.size(0), x_s1_avg.size(1))
        x = self.visible.layer2(x)
        #x_s2 = x
       # x_s2_avg = self.visible.avgpool(x_s2)
        #x_s2_f = x_s2_avg.view(x_s2_avg.size(0), x_s2_avg.size(1))
        x = self.visible.layer3(x)
        x_s3 = x
        
        if self.gm_pool  == 'on':
            b, c, h, w = x_s3.shape
            x_s3_f = x_s3.view(b, c, -1)
            p = 3.0
            x_s3_f = (torch.mean(x_s3_f**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_s3_f = self.visible.avgpool(x_s3)
            x_s3_f = x_s3_f.view(x_s3_f.size(0), x_s3_f.size(1))
        
        x = self.visible.layer4(x)
        x_s4 = x
        if self.gm_pool  == 'on':
            b, c, h, w = x_s4.shape
            x_s4_f = x_s4.view(b, c, -1)
            p = 3.0
            x_s4_f  = (torch.mean(x_s4_f**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_s4_f  = self.visible.avgpool(x_s4)
            x_s4_f  = x_s4_f.view(x_s4_f .size(0),x_s4_f.size(1))
      
        '''
        num_part = 6
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part - 1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        #x = self.visible.avgpool(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        # x = self.dropout(x)
        '''
        return x_s3,x_s3_f,x_s4,x_s4_f
        
class thermal_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18',gm_pool = 'on'):
        super(thermal_net_resnet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)

        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.thermal = model_ft
        self.dropout = nn.Dropout(p=0.5)
        self.gm_pool = gm_pool
    def forward(self, x):
        #pdb.set_trace()
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
       # x_s1 = x
        #x_s1_avg = self.thermal.avgpool(x_s1)
       # x_s1_f = x_s1_avg.view(x_s1_avg.size(0), x_s1_avg.size(1))
        x = self.thermal.layer2(x)
       # x_s2 = x
        #x_s2_avg = self.thermal.avgpool(x_s2)
        #x_s2_f = x_s2_avg.view(x_s2_avg.size(0), x_s2_avg.size(1))
        x = self.thermal.layer3(x)
        x_s3= x
        if self.gm_pool == 'on':
            b, c, h, w = x_s3.shape
            x_s3_f = x_s3.view(b, c, -1)
            p = 3.0
            x_s3_f = (torch.mean(x_s3_f**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_s3_f = self.visible.avgpool(x_s3)
            x_s3_f = x_s3_f.view(x_s3_f.size(0), x_s3_f.size(1))
        x = self.thermal.layer4(x)
        x_s4 = x
        if self.gm_pool  == 'on':
            b, c, h, w = x_s4.shape
            x_s4_f = x_s4.view(b, c, -1)
            p = 3.0
            x_s4_f  = (torch.mean(x_s4_f**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_s4_f  = self.visible.avgpool(x_s4)
            x_s4_f  = x_s4_f.view(x_s4_f .size(0),x_s4_f.size(1))
        '''
        num_part = 6 # number of part
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part-1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        #x = self.thermal.avgpool(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        # x = self.dropout(x)
        '''
        return x_s3,x_s3_f,x_s4,x_s4_f
class num_part(nn.Module):
    def __init__(self,num):
        super(num_part, self).__init__() 
        self.num = num
    def forward(self, x):
        sx = x.size(2) /(self.num)
        sx = int(sx)
        kx = x.size(2) - sx * ((self.num)-1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        x = x.view(x.size(0), x.size(1), x.size(2))
        return x         
class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop = 0.5, arch ='resnet50',gm_pool = 'on'):
        super(embed_net, self).__init__()
        if arch =='resnet18':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 512
        elif arch =='resnet50':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 2048

        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature3 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature4 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature5 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature6 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.classifier1 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier6 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier = ClassBlock(low_dim, class_num, dropout=drop)

        self.l2norm = Normalize(2)
        self.id_fc1 = nn.Linear(2048, 64)
        self.id_fc2 = nn.Linear(2048, 64)
        self.id_tanh = nn.Tanh()
        self.conv_basis= ConvBasis2d(64,1024, 512, kernel_size=1, padding=0, bias=False)
        #self.conv_basis_i= ConvBasis2d(1024,1024, 512, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        #self.fc_output = nn.Linear(1024,512)
        self.num_part = num_part(6)
        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.Conv1x1_r = Conv1x1(1024,512)
        self.Conv1x1_i = Conv1x1(1024,512)
        
    def forward(self, x1, x2, modal = 0 ):
        if modal==0:
            #pdb.set_trace()
            x1_s3,x1_s3_f,x1_s4 ,x1_s4_f= self.visible_net(x1)
            x2_s3,x2_s3_f,x2_s4,x2_s4_f= self.thermal_net(x2)
            
            x1_s4_f_new= self.id_fc1(x1_s4_f)
            x1_s4_f_new = self.id_tanh(x1_s4_f_new)
            x_s3_r = self.conv_basis(x1_s3, x1_s4_f_new)
            x1_s3 = self.Conv1x1_r(x1_s3)
            x_s3_r = x_s3_r+x1_s3
            x_s3_r = self.relu(x_s3_r)
            x_s3_r = self.avgpool(x_s3_r)
            x_s3_r = x_s3_r.view(x_s3_r.size(0), -1)
            x_s3_r_f = x_s3_r
            
            x2_s4_f_new = self.id_fc2(x2_s4_f)
            x2_s4_f_new = self.id_tanh(x2_s4_f_new)
            x_s3_i = self.conv_basis(x2_s3, x2_s4_f_new)
            x2_s3 = self.Conv1x1_i(x2_s3)
            x_s3_i = x_s3_i+x2_s3
            x_s3_i = self.relu(x_s3_i)
            x_s3_i = self.avgpool(x_s3_i)
            x_s3_i = x_s3_i.view(x_s3_i.size(0), -1)
            x_s3_i_f = x_s3_i
            
            
            #x2_s3_f = self.fc_output(x2_s3_f)
            x_s3_f = torch.cat((x_s3_r_f,x_s3_i_f),0)
            '''
            x_s2_new = self.conv_basis2(x_s2, x_s4_f)
            x_s2_new = self.relu(x_s2_new)
            x_s2_new = self.avgpool(x_s2_new)
            x_s2_new = x_s2_new.view(x_s2_new.size(0), -1)
            x_s2_f = self.fc_output(x_s2_new)
            
            x_s3_new = self.conv_basis3(x_s3, x_s4_f)
            x_s3_new = self.relu(x_s3_new)
            x_s3_new = self.avgpool(x_s3_new)
            x_s3_new = x_s3_new.view(x_s3_new.size(0), -1)
            x_s3_f = self.fc_output(x_s3_new)
            '''
            #x_f = torch.cat((x_s1_f, x_s4_f), 0)
            x1 = self.num_part(x1_s4)
            x2 = self.num_part(x2_s4)
           
        # x = self.dropout(x)
            x1 = x1.chunk(6,2)
            x1_0 = x1[0].contiguous().view(x1[0].size(0),-1)
            x1_1 = x1[1].contiguous().view(x1[1].size(0), -1)
            x1_2 = x1[2].contiguous().view(x1[2].size(0), -1)
            x1_3 = x1[3].contiguous().view(x1[3].size(0), -1)
            x1_4 = x1[4].contiguous().view(x1[4].size(0), -1)
            x1_5 = x1[5].contiguous().view(x1[5].size(0), -1)
           
            x2 = x2.chunk(6, 2)
            x2_0 = x2[0].contiguous().view(x2[0].size(0), -1)
            x2_1 = x2[1].contiguous().view(x2[1].size(0), -1)
            x2_2 = x2[2].contiguous().view(x2[2].size(0), -1)
            x2_3 = x2[3].contiguous().view(x2[3].size(0), -1)
            x2_4 = x2[4].contiguous().view(x2[4].size(0), -1)
            x2_5 = x2[5].contiguous().view(x2[5].size(0), -1)
            x_0 = torch.cat((x1_0, x2_0), 0)
            x_1 = torch.cat((x1_1, x2_1), 0)
            x_2 = torch.cat((x1_2, x2_2), 0)
            x_3 = torch.cat((x1_3, x2_3), 0)
            x_4 = torch.cat((x1_4, x2_4), 0)
            x_5 = torch.cat((x1_5, x2_5), 0)
        elif modal ==1:
            #pdb.set_trace()
            x_s3,x_s3_f,x_s4,x_s4_f = self.visible_net(x1)
            
            x_s4_f_new = self.id_fc1(x_s4_f)
            x_s4_f_new = self.id_tanh(x_s4_f_new)
            
            x_s3_d43 = self.conv_basis(x_s3, x_s4_f_new)
            
            x_s3 = self.Conv1x1_r(x_s3)
            x_s3_d43 = x_s3_d43+x_s3
            x_s3_d43 = self.relu(x_s3_d43)
            x_s3_d43 = self.avgpool(x_s3_d43)
            x_s3_f = x_s3_d43.view(x_s3_d43.size(0), -1)
           
        
            x  = self.num_part(x_s4)
            x = x.chunk(6,2)
            x_0 = x[0].contiguous().view(x[0].size(0),-1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)
        elif modal ==2:
            x_s3,x_s3_f,x_s4,x_s4_f = self.thermal_net(x2)
            x_s4_f_new = self.id_fc2(x_s4_f)
            x_s4_f_new = self.id_tanh(x_s4_f_new)
            
            x_s3_d43 = self.conv_basis(x_s3, x_s4_f_new)
            
            x_s3 = self.Conv1x1_i(x_s3)
            x_s3_d43 = x_s3_d43+x_s3
            x_s3_d43 = self.relu(x_s3_d43)
            x_s3_d43 = self.avgpool(x_s3_d43)
            x_s3_f = x_s3_d43.view(x_s3_d43.size(0), -1)
            
            x = self.num_part(x_s4)
            x = x.chunk(6, 2)
            x_0 = x[0].contiguous().view(x[0].size(0), -1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)

        y_0 = self.feature1(x_0)
        y_1 = self.feature2(x_1)
        y_2 = self.feature3(x_2)
        y_3 = self.feature4(x_3)
        y_4 = self.feature5(x_4)
        y_5 = self.feature6(x_5)
        y_s3 = x_s3_f
        #y = self.feature(x)
        out_0 = self.classifier1(y_0)
        out_1 = self.classifier2(y_1)
        out_2 = self.classifier3(y_2)
        out_3 = self.classifier4(y_3)
        out_4 = self.classifier5(y_4)
        out_5 = self.classifier6(y_5)
        out_s3 = self.classifier(y_s3)
        #out = self.classifier(y)
        if self.training:
            return (out_0, out_1, out_2, out_3, out_4, out_5,out_s3), (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4), self.l2norm(y_5),self.l2norm(y_s3))
        else:
            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)
            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            y_4 = self.l2norm(y_4)
            y_5 = self.l2norm(y_5)
            y_s3 = self.l2norm(y_s3)
            y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5,y_s3), 1)
            return  y

            
# debug model structure

# net = embed_net(512, 319)
# net.train()
# input = Variable(torch.FloatTensor(8, 3, 224, 224))
# x, y  = net(input, input)