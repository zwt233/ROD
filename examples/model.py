import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *

class ROD_cluster(nn.Module):
    def __init__(self, dims, n_clusters, num_hops):
        super(ROD_cluster, self).__init__()
        self.num_hops = num_hops + 1
        self.fcs = nn.ModuleList()
        for _ in range(self.num_hops):
            self.fcs.append(nn.Linear(dims[0], dims[1]))

        self.lr_att1 = nn.Linear(dims[0], 1)
        self.lr_att2 = nn.Linear(n_clusters, 1)

        self.dcs = SampleDecoder(act=lambda x: x)

    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std

        return z_scaled

    def forward(self, x):
        attention_scores = [torch.sigmoid(self.lr_att1(i)).view(x[0].shape[0], 1) for i in x]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)

        right_1 = torch.mul(x[0], W[:, 0].view(x[0].shape[0], 1))
        for i in range(1, self.num_hops):
            right_1 = right_1 + torch.mul(x[i], W[:, i].view(x[0].shape[0], 1))

        out_list = []
        x = right_1
        for i in range(self.num_hops):
            out = self.fcs[i](x)
            out = self.scale(out)
            out = F.normalize(out)
            out_list.append(out)

        return out_list
    

class ROD_lp(nn.Module):
    def __init__(self, dims, num_hops):
        super(ROD_lp, self).__init__()
        self.num_hops = num_hops + 1
        self.fcs = nn.ModuleList()
        for _ in range(self.num_hops):
            self.fcs.append(nn.Linear(dims[0], dims[1]))

        self.lr_att1 = nn.Linear(dims[0], 1)
        self.lr_att2 = nn.Linear(1, 1)

        self.dcs = SampleDecoder(act=lambda x: x)

    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std

        return z_scaled

    def forward(self, x):
        attention_scores = [torch.sigmoid(self.lr_att1(i)).view(x[0].shape[0], 1) for i in x]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)

        right_1 = torch.mul(x[0], W[:, 0].view(x[0].shape[0], 1))
        for i in range(1, self.num_hops):
            right_1 = right_1 + torch.mul(x[i], W[:, i].view(x[0].shape[0], 1))

        out_list = []
        x = right_1
        for i in range(self.num_hops):
            out = self.fcs[i](x)
            out = self.scale(out)
            out = F.normalize(out)
            out_list.append(out)

        return out_list


class ROD_cla(nn.Module):
    def __init__(self, nfeat,nhid, nclass, dropout, num_hops):
        super(ROD_cla, self).__init__()
        self.num_hops = num_hops + 1
        self.lr1 = nn.Linear(nfeat, nhid)
        self.lrs = nn.ModuleList()
        for _ in range(self.num_hops):
            self.lrs.append(nn.Linear(nhid, nclass))

        self.lr_att = nn.Linear(nfeat, 1)
        self.dropout = dropout

    def forward(self, feature_list):     
        num_node = feature_list[0].shape[0]
        drop_features = [F.dropout(feature, self.dropout, training=self.training) for feature in feature_list]

        attention_scores = [torch.sigmoid(self.lr_att(x)).view(num_node,1) for x in drop_features]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W,1)
        
        x = torch.mul(drop_features[0], W[:,0].view(num_node,1)) 
        for i in range(1, self.num_hops):
            x = x + torch.mul(drop_features[i], W[:,i].view(num_node,1)) 
            
        x = F.relu(self.lr1(x))
        x = F.dropout(x, self.dropout, training=self.training)

        out_list = []
        for i in range(self.num_hops):
            out = self.lrs[i](x)
            out_list.append(out)
        
        return out_list