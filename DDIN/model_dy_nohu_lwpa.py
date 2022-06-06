import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from LWPA import IWPA
from Dynamic_conv import ConvBasis2d
import torch.nn.functional as F
import pdb
import cv2
import os
import numpy as np
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
            
class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""
    
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
        
# Define the ResNet18-based Model
class visible_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18'):
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
    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        s1 = x
        x = self.visible.layer2(x)
        s2 = x
        x = self.visible.layer3(x)
        s3 = x
        s3_f = F.adaptive_avg_pool2d(s3,(1,1))
        s3_f = s3_f.view(s3_f.size(0), s3_f.size(1))
        
        x = self.visible.layer4(x)
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
        return x,s1,s2,s3,s3_f
        
class thermal_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18'):
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
    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        s1 = x
        x = self.thermal.layer2(x)
        s2 = x
        x = self.thermal.layer3(x)
        s3 = x
        s3_f = F.adaptive_avg_pool2d(s3,(1,1))
        s3_f = s3_f.view(s3_f.size(0), s3_f.size(1))
        
        x = self.thermal.layer4(x)
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
        return x,s1,s2,s3,s3_f
  
class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop = 0.5, arch ='resnet50',lma = True):
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
        self.feature = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.features3 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.classifier1 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier6 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifiers3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.fc = FeatureBlock(1024, 2048, dropout=drop)
        self.fc_s3 = FeatureBlock(1024,2048,dropout=drop)
        self.id_fc1 = FeatureBlock(1024, 64, dropout=drop)
        self.id_fc2 = FeatureBlock(1024, 64, dropout=drop)
        self.id_tanh = nn.Tanh()
        self.conv_basis= ConvBasis2d(64,1024,1024, kernel_size=1, padding=0, bias=False)
        #self.Conv1x1_r = Conv1x1(1024,512)
        #self.Conv1x1_i = Conv1x1(1024,512)
        self.relu = nn.ReLU(inplace=True)
    

        self.l2norm = Normalize(2)
        self.convs1 = Conv1x1(256,1024)
        self.convs2 = Conv1x1(512,1024)
        self.lma = lma
        self.wpa = IWPA(1024, 3)
    
    def forward(self, x1, x2, modal = 0 ):
        if modal==0:
            #pdb.set_trace()
            x1,x1_s1,x1_s2,x1_s3,x1_s3_f= self.visible_net(x1)
            #dynamic
            r_s3_d = x1_s3
            
            #LWPA
            r_s3 = x1_s3
            r_s3 = F.adaptive_avg_pool2d(r_s3, (1, 1))
            r_s3 = r_s3.view(r_s3.size(0), r_s3.size(1))
            
            x1_s1 = self.convs1(x1_s1)
            x1_s1 = F.adaptive_avg_pool2d(x1_s1, (1, 1))
            
            x1_s2 = self.convs2(x1_s2)
            x1_s2 = F.adaptive_avg_pool2d(x1_s2, (1, 1))
            
            x1_s3 = F.adaptive_avg_pool2d(x1_s3, (1, 1))
            x_r = torch.cat((x1_s1,x1_s2,x1_s3),0)
            
            x1 = x1.chunk(6,2)
            x1_0 = x1[0].contiguous().view(x1[0].size(0),-1)
            x1_1 = x1[1].contiguous().view(x1[1].size(0), -1)
            x1_2 = x1[2].contiguous().view(x1[2].size(0), -1)
            x1_3 = x1[3].contiguous().view(x1[3].size(0), -1)
            x1_4 = x1[4].contiguous().view(x1[4].size(0), -1)
            x1_5 = x1[5].contiguous().view(x1[5].size(0), -1)
            x2,x2_s1,x2_s2,x2_s3,x2_s3_f = self.thermal_net(x2)
            #dynamic
            i_s3_d = x2_s3
            
            #LWPA
            i_s3 = x2_s3
            i_s3 = F.adaptive_avg_pool2d(i_s3, (1, 1))
            i_s3 = i_s3.view(i_s3.size(0), i_s3.size(1))
            
            x2_s1 = self.convs1(x2_s1)
            x2_s1 = F.adaptive_avg_pool2d(x2_s1, (1, 1))
            
            x2_s2 = self.convs2(x2_s2)
            x2_s2 = F.adaptive_avg_pool2d(x2_s2, (1, 1))
            
            x2_s3 = F.adaptive_avg_pool2d(x2_s3, (1, 1))
            x_i = torch.cat((x2_s1,x2_s2,x2_s3),0)
            
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
            
            
            #dynamic
            #pdb.set_trace()
            x1_s3_f_new= self.id_fc1(x1_s3_f)
            x1_s3_f_new = self.id_tanh(x1_s3_f_new)
            x_s3_r_local = self.conv_basis(r_s3_d, x1_s3_f_new)
            #r_s3_d = F.adaptive_avg_pool2d(r_s3_d,(16,7))
            #r_s3_d = x_s3_r+r_s3_d
            #r_s3_d = x_s3_r+i_s3_d
            #r_s3_d = self.relu(r_s3_d)
            #r_s3_d = F.adaptive_avg_pool2d(r_s3_d,(1,1))
            #r_s3_d = r_s3_d.view(r_s3_d.size(0), -1)
            #r_s3_d_f = r_s3_d
            
            x2_s3_f_new = self.id_fc2(x2_s3_f)
            x2_s3_f_new = self.id_tanh(x2_s3_f_new)
            x_s3_i_local = self.conv_basis(i_s3_d, x2_s3_f_new)
            #i_s3_d = F.adaptive_avg_pool2d(i_s3_d,(16,7))
            #i_s3_d = x_s3_i+i_s3_d
            '''
            i_s3_d = x_s3_i+r_s3_d
            i_s3_d = self.relu(i_s3_d)
            i_s3_d = F.adaptive_avg_pool2d(i_s3_d,(1,1))
            i_s3_d = i_s3_d.view(i_s3_d.size(0), -1)
            i_s3_d_f = i_s3_d
            x_s3_f = torch.cat((r_s3_d_f,i_s3_d_f),0)
            x_s3_f = self.fc_s3(x_s3_f)
            '''
            #CROSS_noHU
            r_s3_d = r_s3_d + x_s3_r_local
            r_s3_d = self.relu(r_s3_d)
            r_s3_hu = r_s3_d
            r_s3_d = F.adaptive_avg_pool2d(r_s3_d,(1,1))
            r_s3_d = r_s3_d.view(r_s3_d.size(0), -1)
            r_s3_d_f = r_s3_d
            
            i_s3_d = i_s3_d + x_s3_i_local
            i_s3_d = self.relu(i_s3_d)
            i_s3_hu = i_s3_d
            i_s3_d = F.adaptive_avg_pool2d(i_s3_d,(1,1))
            i_s3_d = i_s3_d.view(i_s3_d.size(0), -1)
            i_s3_d_f = i_s3_d
            
            x_s3_f = torch.cat((r_s3_d_f,i_s3_d_f),0)
            x_s3_f = self.fc_s3(x_s3_f)
            
            s3_hu = torch.cat((r_s3_hu, i_s3_hu), 0)
            
            #LWPA
            if self.lma:
              #pdb.set_trace()
              layer_attr = self.wpa(x_r, r_s3, 1, 3)
              layer_atti = self.wpa(x_i, i_s3, 1, 3)
            layer_att = torch.cat((layer_attr, layer_atti), 0)
            layer_att = self.fc(layer_att)
         
           
            
        elif modal ==1:
            #pdb.set_trace()
            x ,x_s1,x_s2,x_s3,x_s3_f= self.visible_net(x1)
            
            r_s3_d = x_s3
            
            s3 = x_s3
            s3 = F.adaptive_avg_pool2d(s3, (1, 1))
            s3 = s3.view(s3.size(0), s3.size(1))
            
            x_s1 = self.convs1(x_s1)
            x_s1 = F.adaptive_avg_pool2d(x_s1, (1, 1))
            
            x_s2 = self.convs2(x_s2)
            x_s2 = F.adaptive_avg_pool2d(x_s2, (1, 1))
            
            x_s3 = F.adaptive_avg_pool2d(x_s3, (1, 1))
            
            s_cat= torch.cat((x_s1,x_s2,x_s3),0)
            
            #LWPA
            if self.lma:
              layer_att = self.wpa(s_cat, s3, 1, 3)
            layer_att = layer_att
            layer_att = self.fc(layer_att)
            
            #dynamic
            x_s3_f_new= self.id_fc1(x_s3_f)
            x_s3_f_new = self.id_tanh(x_s3_f_new)
            x_s3_r_local = self.conv_basis(r_s3_d, x_s3_f_new)
            #r_s3_d = F.adaptive_avg_pool2d(r_s3_d,(16,7))
            r_s3_d = x_s3_r_local+r_s3_d
            r_s3_d = self.relu(r_s3_d)
            s3_hu = r_s3_d
            r_s3_d = F.adaptive_avg_pool2d(r_s3_d,(1,1))
            r_s3_d = r_s3_d.view(r_s3_d.size(0), -1)
            r_s3_d_f = r_s3_d
            x_s3_f = self.fc_s3(r_s3_d_f)
          
            
          
            x = x.chunk(6,2)
            x_0 = x[0].contiguous().view(x[0].size(0),-1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)
        elif modal ==2:
            x,x_s1,x_s2,x_s3,x_s3_f= self.thermal_net(x2)
            
            i_s3_d = x_s3
            
            x = x.chunk(6, 2)
            x_0 = x[0].contiguous().view(x[0].size(0), -1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)
            
            s3 = x_s3
            s3 = F.adaptive_avg_pool2d(s3, (1, 1))
            s3 = s3.view(s3.size(0), s3.size(1))
            
            x_s1 = self.convs1(x_s1)
            x_s1 = F.adaptive_avg_pool2d(x_s1, (1, 1))
            
            x_s2 = self.convs2(x_s2)
            x_s2 = F.adaptive_avg_pool2d(x_s2, (1, 1))
            
            x_s3 = F.adaptive_avg_pool2d(x_s3, (1, 1))
            
            s_cat= torch.cat((x_s1,x_s2,x_s3),0)
            
            #LWPA
            if self.lma:
              layer_att = self.wpa(s_cat, s3, 1, 3)
            layer_att = layer_att
            layer_att = self.fc(layer_att)
            
            #dynamic
            x_s3_f_new = self.id_fc2(x_s3_f)
            x_s3_f_new = self.id_tanh(x_s3_f_new)
            x_s3_i_local = self.conv_basis(i_s3_d, x_s3_f_new)
            #i_s3_d = F.adaptive_avg_pool2d(i_s3_d,(16,7))
            i_s3_d = x_s3_i_local+i_s3_d
            i_s3_d = self.relu(i_s3_d)
            s3_hu = i_s3_d
            i_s3_d = F.adaptive_avg_pool2d(i_s3_d,(1,1))
            i_s3_d = i_s3_d.view(i_s3_d.size(0), -1)
            i_s3_d_f = i_s3_d
            
            x_s3_f = self.fc_s3(i_s3_d_f)
        
        #view = torch_vis_color(s3_hu,18,9,'/DATA/fengmengya/save_image/')
        y_0 = self.feature1(x_0)
        y_1 = self.feature2(x_1)
        y_2 = self.feature3(x_2)
        y_3 = self.feature4(x_3)
        y_4 = self.feature5(x_4)
        y_5 = self.feature6(x_5)
        #y_feat = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5),0)
        y_layer_att = self.feature(layer_att)
        y_s3 = self.features3(x_s3_f)
        
        out_0 = self.classifier1(y_0)
        out_1 = self.classifier2(y_1)
        out_2 = self.classifier3(y_2)
        out_3 = self.classifier4(y_3)
        out_4 = self.classifier5(y_4)
        out_5 = self.classifier6(y_5)
        out_layer_att = self.classifier(y_layer_att)
        out_s3 = self.classifiers3(y_s3)
        
        if self.training:
            return (out_0, out_1, out_2, out_3, out_4, out_5,out_layer_att,out_s3), (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4), self.l2norm(y_5),self.l2norm(y_layer_att),self.l2norm(y_s3))
        else:
            x_0 = self.l2norm(x_0)
            x_1 = self.l2norm(x_1)
            x_2 = self.l2norm(x_2)
            x_3 = self.l2norm(x_3)
            x_4 = self.l2norm(x_4)
            x_5 = self.l2norm(x_5)#32 2048
            
            layer_att =  self.l2norm(layer_att)
            
            x_s3_f = self.l2norm(x_s3_f)
            
            #x_all_part = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5), 0)
            x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5,layer_att,x_s3_f), 1)
            
            
            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)
            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            y_4 = self.l2norm(y_4)
            y_5 = self.l2norm(y_5)#
            
            #y_all_part = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5), 0)
            
            y_layer_att = self.l2norm(y_layer_att)
            
            y_s3 = self.l2norm(y_s3)
            
            y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5,y_layer_att,y_s3), 1)
            
            return x, y,s3_hu

            