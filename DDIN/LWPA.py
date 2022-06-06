import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
        
class IWPA(nn.Module):
    def __init__(self, in_channels, layer = 3, inter_channels=None, out_channels=None):
        super(IWPA, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.l2norm = Normalize(2)

        if self.inter_channels is None:
            self.inter_channels = in_channels

        if self.out_channels is None:
            self.out_channels = in_channels

        conv_nd = nn.Conv2d

        self.fc1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.fc2 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.fc3 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.out_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)


        self.bottleneck = nn.BatchNorm1d(in_channels)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias.data)

        # weighting vector of the part features
        self.gate = nn.Parameter(torch.FloatTensor(layer))
        nn.init.constant_(self.gate, 1/layer)
    def forward(self, layer, feat, t=None, part=0):
        #pdb.set_trace()
        bt, c, h, w = layer.shape
        b = bt // 3
        # get part features
        #part_feat = F.adaptive_avg_pool2d(x, (part, 1))#64*2048*3*1
        #part_feat = part_feat.view(b, t, c, part)#64*1*2048*3
        layer_feat = layer.permute(0, 1, 3, 2) # B, C, T, Part#64*2048*1*3

        layer_feat1 = self.fc1(layer_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part
        layer_feat1 = layer_feat1.permute(0, 2, 1)  # B, T*Part, C//r

        layer_feat2 = self.fc2(layer_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part

        layer_feat3 = self.fc3(layer_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part
        layer_feat3 = layer_feat3.permute(0, 2, 1)   # B, T*Part, C//r

        # get cross-part attention
        cpa_att = torch.matmul(layer_feat1, layer_feat2) # B, T*Part, T*Part
        cpa_att = F.softmax(cpa_att, dim=-1)

        # collect contextual information
        refined_layer_feat = torch.matmul(cpa_att, layer_feat3) # B, T*Part, C//r
        refined_layer_feat = refined_layer_feat.permute(0, 2, 1).contiguous() # B, C//r, T*Part
        refined_layer_feat = refined_layer_feat.view(b, self.inter_channels, 3) # B, C//r, T, Part

        gate = F.softmax(self.gate, dim=-1)
        weight_layer_feat = torch.matmul(refined_layer_feat, gate)
        #x = F.adaptive_avg_pool2d(x, (1, 1))
        # weight_part_feat = weight_part_feat + x.view(x.size(0), x.size(1))

        weight_layer_feat = weight_layer_feat+ feat
        feat = self.bottleneck(weight_layer_feat)

        return feat