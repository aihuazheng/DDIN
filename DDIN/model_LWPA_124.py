import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from LWPA import IWPA
import torch.nn.functional as F
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
        x = self.visible.layer4(x)
        s4 = x
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
        return x,s1,s2,s3,s4
        
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
        x = self.thermal.layer4(x)
        s4 = x
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
        return x,s1,s2,s3,s4
        
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
        self.classifier1 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier6 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier = ClassBlock(low_dim, class_num, dropout=drop)
        #self.fc = FeatureBlock(1024, 2048, dropout=drop)
        
    

        self.l2norm = Normalize(2)
        self.convs1 = Conv1x1(256,2048)
        self.convs2 = Conv1x1(512,2048)
        self.convs3 = Conv1x1(1024,2048)
        self.lma = lma
        self.wpa = IWPA(2048, 3)

    def forward(self, x1, x2, modal = 0 ):
        if modal==0:
            #pdb.set_trace()
            x1,x1_s1,x1_s2,x1_s3,x1_s4= self.visible_net(x1)
            r_s3 = x1_s3
            r_s3 = self.convs3(r_s3)
            r_s3 = F.adaptive_avg_pool2d(r_s3, (1, 1))
            r_s3 = r_s3.view(r_s3.size(0), r_s3.size(1))
            
            r_s4 = x1_s4
            r_s4 = F.adaptive_avg_pool2d(r_s4, (1, 1))
            r_s4 = r_s4.view(r_s4.size(0), r_s4.size(1))
            
            x1_s1 = self.convs1(x1_s1)
            x1_s1 = F.adaptive_avg_pool2d(x1_s1, (1, 1))
            
            x1_s2 = self.convs2(x1_s2)
            x1_s2 = F.adaptive_avg_pool2d(x1_s2, (1, 1))
            
            x1_s3 = self.convs3(x1_s3)
            x1_s3 = F.adaptive_avg_pool2d(x1_s3, (1, 1))
            
            x1_s4 = F.adaptive_avg_pool2d(x1_s4, (1, 1))
            
            #x_r_123 = torch.cat((x1_s1,x1_s2,x1_s3),0)
            x_r_124 = torch.cat((x1_s1,x1_s2,x1_s4),0)
            #x_r_134 = torch.cat((x1_s1,x1_s3,x1_s4),0)
            #x_r_234 = torch.cat((x1_s2,x1_s3,x1_s4),0)
            
            x1 = x1.chunk(6,2)
            x1_0 = x1[0].contiguous().view(x1[0].size(0),-1)
            x1_1 = x1[1].contiguous().view(x1[1].size(0), -1)
            x1_2 = x1[2].contiguous().view(x1[2].size(0), -1)
            x1_3 = x1[3].contiguous().view(x1[3].size(0), -1)
            x1_4 = x1[4].contiguous().view(x1[4].size(0), -1)
            x1_5 = x1[5].contiguous().view(x1[5].size(0), -1)
            
            x2,x2_s1,x2_s2,x2_s3,x2_s4 = self.thermal_net(x2)
           
            i_s3 = x2_s3
            i_s3 = self.convs3(i_s3)
            i_s3 = F.adaptive_avg_pool2d(i_s3, (1, 1))
            i_s3 = i_s3.view(i_s3.size(0), i_s3.size(1))
            
            i_s4 = x2_s4
            i_s4 = F.adaptive_avg_pool2d(i_s4, (1, 1))
            i_s4 = i_s4.view(i_s4.size(0), i_s4.size(1))
            
            x2_s1 = self.convs1(x2_s1)
            x2_s1 = F.adaptive_avg_pool2d(x2_s1, (1, 1))
            
            x2_s2 = self.convs2(x2_s2)
            x2_s2 = F.adaptive_avg_pool2d(x2_s2, (1, 1))
            
            x2_s3 = self.convs3(x2_s3)
            x2_s3 = F.adaptive_avg_pool2d(x2_s3, (1, 1))
            
            x2_s4 = F.adaptive_avg_pool2d(x2_s4, (1, 1))
            
            #x_i_123 = torch.cat((x2_s1,x2_s2,x2_s3),0)
            x_i_124 = torch.cat((x2_s1,x2_s2,x2_s4),0)
            #x_i_134 = torch.cat((x2_s1,x2_s3,x2_s4),0)
            #x_i_234 = torch.cat((x2_s2,x2_s3,x2_s4),0)
            
            
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
            
            if self.lma:
              #pdb.set_trace()
              layer_attr_124 = self.wpa(x_r_124, r_s4, 1, 3)
              layer_atti_124 = self.wpa(x_i_124, i_s4, 1, 3)
              
            layer_att = torch.cat((layer_attr_124, layer_atti_124), 0)
            #s4 = torch.cat((r_s4, i_s4), 0)
            
            #layer_att = torch.cat((layer_att_123,s4), 1)
             
        elif modal ==1:
            #pdb.set_trace()
            x ,x_s1,x_s2,x_s3,x_s4= self.visible_net(x1)
            
            s3 = x_s3
            s3 = self.convs3(s3)
            s3 = F.adaptive_avg_pool2d(s3, (1, 1))
            s3 = s3.view(s3.size(0), s3.size(1))
            
            s4 = x_s4
            s4 = F.adaptive_avg_pool2d(s4, (1, 1))
            s4 = s4.view(s4.size(0), s4.size(1))
            
            x_s1 = self.convs1(x_s1)
            x_s1 = F.adaptive_avg_pool2d(x_s1, (1, 1))
            
            x_s2 = self.convs2(x_s2)
            x_s2 = F.adaptive_avg_pool2d(x_s2, (1, 1))
            
            x_s3 = self.convs3(x_s3)
            x_s3 = F.adaptive_avg_pool2d(x_s3, (1, 1))
            
            x_s4 = F.adaptive_avg_pool2d(x_s4, (1, 1))
            
            
            s_cat_124= torch.cat((x_s1,x_s2,x_s4),0)
        
            
            if self.lma:
              layer_att_124 = self.wpa(s_cat_124, s4, 1, 3)
             
              
            layer_att = layer_att_124
          
            x = x.chunk(6,2)
            x_0 = x[0].contiguous().view(x[0].size(0),-1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)
        elif modal ==2:
            x,x_s1,x_s2,x_s3,x_s4 = self.thermal_net(x2)
            x = x.chunk(6, 2)
            x_0 = x[0].contiguous().view(x[0].size(0), -1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)
            
            s3 = x_s3
            s3 = self.convs3(s3)
            s3 = F.adaptive_avg_pool2d(s3, (1, 1))
            s3 = s3.view(s3.size(0), s3.size(1))
            
            s4 = x_s4
            s4 = F.adaptive_avg_pool2d(s4, (1, 1))
            s4 = s4.view(s4.size(0), s4.size(1))
            
            x_s1 = self.convs1(x_s1)
            x_s1 = F.adaptive_avg_pool2d(x_s1, (1, 1))
            
            x_s2 = self.convs2(x_s2)
            x_s2 = F.adaptive_avg_pool2d(x_s2, (1, 1))
            
            x_s3 = self.convs3(x_s3)
            x_s3 = F.adaptive_avg_pool2d(x_s3, (1, 1))
            
            x_s4 = F.adaptive_avg_pool2d(x_s4, (1, 1))
            
            s_cat_124= torch.cat((x_s1,x_s2,x_s4),0)
           
            
            if self.lma:
              layer_att_124 = self.wpa(s_cat_124, s4, 1, 3)
             
              
            layer_att = layer_att_124  
         

        y_0 = self.feature1(x_0)
        y_1 = self.feature2(x_1)
        y_2 = self.feature3(x_2)
        y_3 = self.feature4(x_3)
        y_4 = self.feature5(x_4)
        y_5 = self.feature6(x_5)#64,512
        
        y_layer_att = self.feature(layer_att)#64,512
        
        out_0 = self.classifier1(y_0)
        out_1 = self.classifier2(y_1)
        out_2 = self.classifier3(y_2)
        out_3 = self.classifier4(y_3)
        out_4 = self.classifier5(y_4)
        out_5 = self.classifier6(y_5)#64,395
        out_layer_att = self.classifier(y_layer_att)#64,395
        if self.training:
            return (out_0, out_1, out_2, out_3, out_4, out_5,out_layer_att), (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4), self.l2norm(y_5),self.l2norm(y_layer_att))
        else:
            x_0 = self.l2norm(x_0)
            x_1 = self.l2norm(x_1)
            x_2 = self.l2norm(x_2)
            x_3 = self.l2norm(x_3)
            x_4 = self.l2norm(x_4)
            x_5 = self.l2norm(x_5)#32 2048
            
            layer_att =  self.l2norm(layer_att)#\
            
            x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5,layer_att), 1)
            
            
            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)
            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            y_4 = self.l2norm(y_4)
            y_5 = self.l2norm(y_5)#
            
            y_layer_att = self.l2norm(y_layer_att)#64, 512
            
            y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5,y_layer_att), 1)
            
            return x, y
            
