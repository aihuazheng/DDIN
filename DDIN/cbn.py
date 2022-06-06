import torch
import sys
import torch.nn as nn
import pdb
import torch.nn.functional as F
from torch.autograd import Variable
class CBN(nn.Module):
    def __init__(self,IR_size,emb_size,out_size,batch_size,channels,use_betas=True,use_gammas=True,eps=1.0e-5):
        super(CBN, self).__init__()
        self.IR_size = IR_size
        self.emb_size = emb_size # size of hidden layer of MLP
        self.out_size = out_size # output of the MLP - for each channel
        self.batch_size = batch_size
        self.channels = channels
        self.use_betas = use_betas
        self.use_gammas = use_gammas
        self.softmax = nn.Softmax(dim=1)
        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(self.batch_size, self.channels).cuda())
        self.gammas = nn.Parameter(torch.ones(self.batch_size, self.channels).cuda())
        self.eps = eps
    
        # MLP used to predict betas and gammas

        self.fc_gamma = nn.Sequential(
            nn.Linear(self.IR_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            ).cuda()
            
        self.fc_beta = nn.Sequential(
            nn.Linear(self.IR_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            ).cuda()

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)
    '''
    Predicts the value of delta beta and delta gamma for each channel
    Arguments:
        lstm_emb : lstm embedding of the question
    Returns:
        delta_betas, delta_gammas : for each layer
    '''
    def create_cbn_input(self, IR_emb):
        if self.use_betas:
            delta_betas = self.fc_beta(IR_emb)
        else:
            delta_betas = torch.zeros(self.batch_size, self.channels).cuda()
        if self.use_gammas:

            delta_gammas = self.fc_gamma(IR_emb)
        else:
            delta_gammas = torch.zeros(self.batch_size, self.channels).cuda()
        return delta_betas, delta_gammas
    def forward(self, feature, IR_emb):
        #pdb.set_trace()
        self.batch_size, self.channels, self.height, self.width = feature.data.shape
       # get delta values
        delta_betas, delta_gammas = self.create_cbn_input(IR_emb)
        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()
        # update the values of beta and gamma
        betas_cloned += delta_betas
        gammas_cloned += delta_gammas
        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)
        # extend the betas and gammas of each channel across the height and width of feature map
        betas_expanded = torch.stack([betas_cloned]*self.height, dim=2)
        betas_expanded = torch.stack([betas_expanded]*self.width, dim=3)
        gammas_expanded = torch.stack([gammas_cloned]*self.height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded]*self.width, dim=3)
        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+self.eps)
        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded
        out = torch.mul(out,self.softmax(out))
        return out ,IR_emb


# testing code
'''
if __name__ == '__main__':

    #torch.cuda.set_device(int(sys.argv[0]))

    model = CBN(512, 256,256,8,256).cuda()
    rgb = Variable(torch.FloatTensor(8, 256, 224, 224)).cuda()
    ir = Variable(torch.FloatTensor(8,512)).cuda()
    out,lstm_emb = model(rgb,ir)
    print out 
    print lstm_emb

    for m in model.modules():

        if isinstance(m, nn.BatchNorm2d):

            print ('found anomaly')

        if isinstance(m, nn.Linear):

            print ('found correct')
'''




