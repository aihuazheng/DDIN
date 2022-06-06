import math
import torch
import torch.nn as nn
from torchvision.models import ResNet
import numpy as np
import torch.nn.functional as F
import pdb

class RegularLoss(nn.Module):

    def __init__(self, gamma=0, part_features=None, nparts=1):
        """
        :param bs: batch size
        :param ncrops: number of crops used at constructing dataset
        """
        super(RegularLoss, self).__init__()
        self.register_buffer('part_features', part_features)
        self.nparts = nparts
        self.gamma = gamma
        # self.batchsize = bs
        # self.ncrops = ncrops

    def forward(self, x):
        assert isinstance(x, list), "parts features should be presented in a list"
        corr_matrix = torch.zeros(self.nparts, self.nparts)
        # x = [torch.div(xx, xx.norm(dim=1, keepdim=True)) for xx in x]
        for i in range(self.nparts):
            x[i] = x[i].squeeze()
            x[i] = torch.div(x[i], x[i].norm(dim=1, keepdim=True))

        for i in range(self.nparts):
            for j in range(self.nparts):
                corr_matrix[i, j] = torch.mean(torch.mm(x[i], x[j].t()))
                if i == j:
                    corr_matrix[i, j] = 1.0 - corr_matrix[i, j]

        return torch.mul(torch.sum(torch.triu(corr_matrix)), self.gamma).to(device)