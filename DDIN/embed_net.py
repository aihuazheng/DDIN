class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop = 0.5, arch ='resnet50'):
        super(embed_net, self).__init__()
        if arch =='resnet18':
            
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 512
            
        elif arch =='resnet50':
            self.v_net = visible_net(arch = arch)
            self.t_net = thermal_net(arch = arch)
            self.com_net = com_net(arch = arch)
            pool_dim_1 = 256
            pool_dim_2 = 512
            pool_dim_3 = 1024
            pool_dim_4 = 2048

        self.feature = FeatureBlock(pool_dim_4, low_dim, dropout = drop)   # linear  -->  BN       
        
        self.classifier = ClassBlock(low_dim, class_num, dropout = drop) # leakyReLU  -->  drouput  -->  linear(class_layer)
        
        self.attr_net1 = nn.Sequential(
            nn.Linear(26, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
        
        self.attr_net2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Softplus(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 395),   
        )
        
        self.feature3 = FeatureBlock(1024, 256, dropout = drop)   # linear  -->  BN       
        
        self.classifier3 = ClassBlock(256, class_num, dropout = drop)
        
        '''
        self.attr_net2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            #nn.ReLU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Softplus(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 395),   
        )
        '''
        self.conv1 = nn.Conv2d(1024, 1024*64, kernel_size=1, stride=1)
        '''
        self.attention_net = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 2048),
            nn.Sigmoid(),
        )
        '''
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.sigmoid = nn.Sigmoid()
        #self.feature_sample = FeatureBlock(384, low_dim, dropout = drop)   # linear  -->  BN       
        
        self.l2norm = Normalize(2)   # L2_normalization
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x1, x2, attributes_labels, modal = 0 ):
        if modal==0:
            
            
            v_input  = self.v_net(x1)
            ir_input = self.t_net(x2)
            
            if self.training:
              
              idfcn = 64
              attr1 = self.attr_net1(attributes_labels)
              
              
              attr2 = attr1.view(-1, 1024, 1, 1)
              #attr2 = attr1.view(-1, 1024, 1, 1)
              attr2 = attr2.expand(v_input.size(0), 1024, v_input.size(2), v_input.size(3))
              #attr2 = attr2.expand(v.size(0), 1024, v.size(2), v.size(3))
              #attr2 = attr2.expand(v_input.size(0), idfcn, 1024, v_input.size(2), v_input.size(3))
              
              attr2 = self.conv1(attr2)
              attr2 = attr2.view(attr2.size(0), idfcn, 1024, attr2.size(2), attr2.size(3))
              
              #v = self.conv1(v_input)
              #v = v.view(v_input.size(0), idfcn, 1024, v_input.size(2), v_input.size(3))
              #v = v * attr2
              
              v = v_input.view(v_input.size(0), 1, 1024, v_input.size(2), v_input.size(3)).expand(v_input.size(0), idfcn, 1024, v_input.size(2), v_input.size(3))
              v = v * attr2
              
              v = v.sum(1).view(attr2.size(0), attr2.size(2), attr2.size(3), attr2.size(4))
              
              #v = self.conv1(v_input)
              #v = v.view(v_input.size(0), idfcn, 1024, v_input.size(2), v_input.size(3))
              #v = v * attr2
              #v = v.sum(1).view(attr2.size(0), attr2.size(2), attr2.size(3), attr2.size(4))
              
              ir = ir_input.view(ir_input.size(0), 1, 1024, ir_input.size(2), ir_input.size(3)).expand(ir_input.size(0), idfcn, 1024, ir_input.size(2), ir_input.size(3))
              ir = ir * attr2
              
              ir = ir.sum(1).view(attr2.size(0), attr2.size(2), attr2.size(3), attr2.size(4))
              
            v = v + v_input
            ir = ir + ir_input
            
            s3_v_ir = torch.cat((v, ir), 0)
            s3_v_ir = self.pool(s3_v_ir).view(s3_v_ir.size(0), s3_v_ir.size(1))
            
            logits3 = self.classifier3(self.feature3(s3_v_ir))
            
            v_feat = self.com_net(v_input, attributes_labels)
            ir_feat = self.com_net(ir_input, attributes_labels)
            
            attr = self.attr_net2(attr1)
            #atten = self.attention_net(attr)
            
            #v_feat = v_feat + v_feat * atten
            #ir_feat = ir_feat + ir_feat * atten
                        
            #g_feat = torch.cat((v_feat, ir_feat), 0)
            g_feat = torch.cat((v_feat, ir_feat), 0)
            
        elif modal ==1:
            
            v  = self.v_net(x1)
            
            v  = self.com_net(v, attributes_labels)
            
            g_feat = v
            
        elif modal ==2:
        
            ir = self.t_net(x2)
            
            ir = self.com_net(ir, attributes_labels)
            
            g_feat = ir
            
# ---------------------------------------------------------------------------------------

        embedding = self.feature(g_feat) 
        #if self.training:
        #  emb = self.feature_sample(labels)
        #  out = self.classifier(emb)
        out3 = self.classifier(embedding)
        #if self.training:
        #  out3 = torch.cat((out, out3), 0)

        if self.training:
            return torch.cat((out3, attr), 0), logits3
            #return out3
        else:
            return self.l2norm(embedding)