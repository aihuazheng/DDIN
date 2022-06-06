import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
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

# Define the ResNet18-based Model
class visible_net_resnet(nn.Module):
    def __init__(self,block,fc_layer1=1024, fc_layer2=512, arch ='resnet18', global_pooling_size=(8,4),drop_rate=0.5,):
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
        self.avgpool1 = nn.AvgPool2d([x*8 for x in global_pooling_size])
        self.avgpool2 = nn.AvgPool2d([x*4 for x in global_pooling_size])
        self.avgpool3 = nn.AvgPool2d([x*2 for x in global_pooling_size])
        self.avgpool4 = nn.AvgPool2d([x*1 for x in global_pooling_size])
        
        self.layer1_fc = nn.Sequential(
            nn.Linear(256, fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer2_fc = nn.Sequential(
            nn.Linear(512 , fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer3_fc = nn.Sequential(
            nn.Linear(1024 , fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer4_fc = nn.Sequential(
            nn.Linear(8192, fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.fusion_conv = nn.Conv1d(4,1,kernel_size=1, bias=False)
    def forward(self, x):
       
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x1 = self.visible.layer1(x)
        x2 = self.visible.layer2(x1)
        x3 = self.visible.layer3(x2)
        x4 = self.visible.layer4(x3)
        x4_p = x4
        num_part = 6
        # pool size
        sx = x4_p.size(2) / num_part
        sx = int(sx)
        kx = x4_p.size(2) - sx * (num_part - 1)
        kx = int(kx)
        x4_p = nn.functional.avg_pool2d(x4_p, kernel_size=(kx, x4_p.size(3)), stride=(sx, x4_p.size(3)))
        #x = self.visible.avgpool(x)
        x4_p = x4_p.view(x4_p.size(0), x4_p.size(1), x4_p.size(2))
        # x = self.dropout(x)
        #pdb.set_trace()
        x1 = self.avgpool1(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.layer1_fc(x1)

        x2 = self.avgpool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.layer2_fc(x2)

        x3 = self.avgpool3(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.layer3_fc(x3)

        x4 = self.avgpool4(x4)
        x4 = x4.view(x4.size(0), -1)
        x4 = self.layer4_fc(x4)

        x5 = torch.cat([x1.unsqueeze(dim=1),x2.unsqueeze(dim=1),x3.unsqueeze(dim=1),x4.unsqueeze(dim=1)],dim=1)
        x5 = self.fusion_conv(x5)
        x5 = x5.view(x5.size(0),-1)
        return x4_p,x5
        
class thermal_net_resnet(nn.Module):
     def __init__(self,block,fc_layer1=1024, fc_layer2=512, arch ='resnet18', global_pooling_size=(8,4),drop_rate=0.,):
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
        self.avgpool1 = nn.AvgPool2d([x*8 for x in global_pooling_size])
        self.avgpool2 = nn.AvgPool2d([x*4 for x in global_pooling_size])
        self.avgpool3 = nn.AvgPool2d([x*2 for x in global_pooling_size])
        self.avgpool4 = nn.AvgPool2d([x*1 for x in global_pooling_size])
        #pdb.set_trace()
        self.layer1_fc = nn.Sequential(
            nn.Linear(256, fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer2_fc = nn.Sequential(
            nn.Linear(512, fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer3_fc = nn.Sequential(
            nn.Linear(1024 , fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer4_fc = nn.Sequential(
            nn.Linear(8192 , fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.fusion_conv = nn.Conv1d(4,1,kernel_size=1, bias=False)
     def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x1 = self.thermal.layer1(x)
        x2 = self.thermal.layer2(x1)
        x3 = self.thermal.layer3(x2)
        x4 = self.thermal.layer4(x3)
        x4_p = x4
        num_part = 6 # number of part
        # pool size
        sx = x4_p.size(2) / num_part
        sx = int(sx)
        kx = x4_p.size(2) - sx * (num_part-1)
        kx = int(kx)
        x4_p = nn.functional.avg_pool2d(x4_p, kernel_size=(kx, x4_p.size(3)), stride=(sx, x4_p.size(3)))
        #x = self.thermal.avgpool(x)
        x4_p = x4_p.view(x4_p.size(0), x4_p.size(1), x4_p.size(2))
        # x = self.dropout(x)
        
        x1 = self.avgpool1(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.layer1_fc(x1)

        x2 = self.avgpool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.layer2_fc(x2)

        x3 = self.avgpool3(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.layer3_fc(x3)

        x4 = self.avgpool4(x4)
        x4 = x4.view(x4.size(0), -1)
        x4 = self.layer4_fc(x4)

        x5 = torch.cat([x1.unsqueeze(dim=1),x2.unsqueeze(dim=1),x3.unsqueeze(dim=1),x4.unsqueeze(dim=1)],dim=1)
        x5 = self.fusion_conv(x5)
        x5 = x5.view(x5.size(0),-1)
        
        return x4_p,x5
        
class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop = 0.5, arch ='resnet50'):
        super(embed_net, self).__init__()
        if arch =='resnet18':
            self.visible_net = visible_net_resnet(arch = arch,block = 4)
            self.thermal_net = thermal_net_resnet(arch = arch,block = 4)
            pool_dim = 512
        elif arch =='resnet50':
            self.visible_net = visible_net_resnet(arch = arch,block = 4)
            self.thermal_net = thermal_net_resnet(arch = arch,block = 4)
            pool_dim = 2048

        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature3 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature4 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature5 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature6 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.classifier1 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier6 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier = ClassBlock(low_dim, class_num, dropout=drop)

        self.l2norm = Normalize(2)
        
        self.fc_x5 = nn.Sequential(
            nn.Linear(512 , 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.),
            nn.Linear(1024, 2048),
        )
        
    def forward(self, x1, x2, modal = 0 ):
        if modal==0:
            x4_p_r,x5_r = self.visible_net(x1)
            x4_p_r = x4_p_r.chunk(6,2)
            x4_p_r_0 = x4_p_r[0].contiguous().view(x4_p_r[0].size(0),-1)
            x4_p_r_1 = x4_p_r[1].contiguous().view(x4_p_r[1].size(0), -1)
            x4_p_r_2 = x4_p_r[2].contiguous().view(x4_p_r[2].size(0), -1)
            x4_p_r_3 = x4_p_r[3].contiguous().view(x4_p_r[3].size(0), -1)
            x4_p_r_4 = x4_p_r[4].contiguous().view(x4_p_r[4].size(0), -1)
            x4_p_r_5 = x4_p_r[5].contiguous().view(x4_p_r[5].size(0), -1)
            
            x4_p_i,x5_i = self.thermal_net(x2)
            x4_p_i = x4_p_i.chunk(6, 2)
            x4_p_i_0 = x4_p_i[0].contiguous().view(x4_p_i[0].size(0), -1)
            x4_p_i_1 = x4_p_i[1].contiguous().view(x4_p_i[1].size(0), -1)
            x4_p_i_2 = x4_p_i[2].contiguous().view(x4_p_i[2].size(0), -1)
            x4_p_i_3 = x4_p_i[3].contiguous().view(x4_p_i[3].size(0), -1)
            x4_p_i_4 = x4_p_i[4].contiguous().view(x4_p_i[4].size(0), -1)
            x4_p_i_5 = x4_p_i[5].contiguous().view(x4_p_i[5].size(0), -1)
            xp_0 = torch.cat((x4_p_r_0, x4_p_i_0), 0)
            xp_1 = torch.cat((x4_p_r_1, x4_p_i_1), 0)
            xp_2 = torch.cat((x4_p_r_2, x4_p_i_2), 0)
            xp_3 = torch.cat((x4_p_r_3, x4_p_i_3), 0)
            xp_4 = torch.cat((x4_p_r_4, x4_p_i_4), 0)
            xp_5 = torch.cat((x4_p_r_5, x4_p_i_5), 0)
            
            x5 = torch.cat((x5_r,x5_i),0)
            x5 = self.fc_x5(x5)
        elif modal ==1:
            xp,x5 = self.visible_net(x1)
            xp = xp.chunk(6,2)
            xp_0 = xp[0].contiguous().view(xp[0].size(0),-1)
            xp_1 = xp[1].contiguous().view(xp[1].size(0), -1)
            xp_2 = xp[2].contiguous().view(xp[2].size(0), -1)
            xp_3 = xp[3].contiguous().view(xp[3].size(0), -1)
            xp_4 = xp[4].contiguous().view(xp[4].size(0), -1)
            xp_5 = xp[5].contiguous().view(xp[5].size(0), -1)
            x5 = self.fc_x5(x5)
        elif modal ==2:
            xp,x5 = self.thermal_net(x2)
            xp = xp.chunk(6,2)
            xp_0 = xp[0].contiguous().view(xp[0].size(0),-1)
            xp_1 = xp[1].contiguous().view(xp[1].size(0), -1)
            xp_2 = xp[2].contiguous().view(xp[2].size(0), -1)
            xp_3 = xp[3].contiguous().view(xp[3].size(0), -1)
            xp_4 = xp[4].contiguous().view(xp[4].size(0), -1)
            xp_5 = xp[5].contiguous().view(xp[5].size(0), -1)
            x5 = self.fc_x5(x5)

        y_0 = self.feature1(xp_0)
        y_1 = self.feature2(xp_1)
        y_2 = self.feature3(xp_2)
        y_3 = self.feature4(xp_3)
        y_4 = self.feature5(xp_4)
        y_5 = self.feature6(xp_5)
        y_fusion = self.feature(x5)
       
        #y = self.feature(x)
        out_0 = self.classifier1(y_0)
        out_1 = self.classifier2(y_1)
        out_2 = self.classifier3(y_2)
        out_3 = self.classifier4(y_3)
        out_4 = self.classifier5(y_4)
        out_5 = self.classifier6(y_5)
        out_fusion = self.classifier6(y_fusion)
        #out = self.classifier(y)
        if self.training:
            return (out_0, out_1, out_2, out_3, out_4, out_5,out_fusion), (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4), self.l2norm(y_5),self.l2norm(y_fusion))
        else:
            #pdb.set_trace()
            xp_0 = self.l2norm(xp_0)
            xp_1 = self.l2norm(xp_1)
            xp_2 = self.l2norm(xp_2)
            xp_3 = self.l2norm(xp_3)
            xp_4 = self.l2norm(xp_4)
            xp_5 = self.l2norm(xp_5)
            x5 = self.l2norm(x5)
            x = torch.cat((xp_0,xp_1,xp_2,xp_3, xp_4,xp_5,x5), 1)
            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)
            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            y_4 = self.l2norm(y_4)
            y_5 = self.l2norm(y_5)
            y_fusion = self.l2norm(y_fusion)
            y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5,y_fusion), 1)
            return x, y

            
