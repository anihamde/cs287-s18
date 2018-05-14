import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import h5py

from baseline_model import conv, to_one_hot, RoadmapDataset, Conv1dNorm, LinearNorm

class BassetNormCat(nn.Module):
    def __init__(self, dropout_prob=0.3, gene_drop_lvl=1):
        super(BassetNormCat, self).__init__()
        self.conv1 = Conv1dNorm(4, 300, 19, stride=1, padding=0, weight_norm=False)
        self.conv2 = Conv1dNorm(300, 200, 11, stride=1, padding=0, weight_norm=False)
        self.conv3 = Conv1dNorm(200, 200, 7, stride=1, padding=0, weight_norm=False)
        self.maxpool_4 = nn.MaxPool1d(4,padding=0)
        self.maxpool_3 = nn.MaxPool1d(3,padding=0)
        self.genelinear = LinearNorm(19795, 500, weight_norm=False)
        self.linear1 = LinearNorm(200*13+500, 1000, weight_norm=False)
        self.linear2 = LinearNorm(1000, 1000, weight_norm=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(1000, 1)
        self.gdl = gene_drop_lvl
    def forward(self, x, geneexpr):
        #if sparse_in: # (?, 600, 4)
        #    in_seq = to_one_hot(x, n_dims=4).permute(0,3,1,2).squeeze()
        #else:
        #    in_seq = x.squeeze()
        x = F.pad(x,(9,9))
        out = F.relu(self.conv1(x)) # (?, 4, 580)
        out = self.maxpool_3(out) # (?, 30, 145)
        out = F.pad(out,(5,5))
        out = F.relu(self.conv2(out)) # (?, 300, 140)
        out = self.maxpool_4(out) # (?, 300, 35)
        out = F.pad(out,(3,3))
        out = F.relu(self.conv3(out)) # (?, 500, 32)
        out = F.pad(out,(1,1))
        out = self.maxpool_4(out) # (?, 500, 8)
        out = out.view(-1, 200*13) # (?, 500*8)
        if self.gdl == 0:
            geneexpr = self.dropout(geneexpr)
            geneexpr = F.relu(self.genelinear(geneexpr))
        elif self.gdl == 1:
            geneexpr = F.relu(self.genelinear(geneexpr)) # (?, 500)
            geneexpr = self.dropout(geneexpr)
        elif self.gdl == 2:
            geneexpr = F.normalize(self.genelinear(geneexpr), p=2, dim=1)
        out = torch.cat([out, geneexpr], dim=1) # (?, 200*13+500)
        out = F.relu(self.linear1(out)) # (?, 800)
        out = self.dropout(out)
        out = F.relu(self.linear2(out)) # (?, 800)
        out = self.dropout(out)
        return self.output(out) # (?, 1)
    def clip_norms(self, value):
        key_chain = [ key for key in self.state_dict().keys() if 'weight' in key ]
        for key in key_chain:
            module_list = [self]
            for key_level in key.split('.')[:-1]:
                module_list.append( getattr(module_list[-1], key_level) )
            if key.split('.')[-1] == 'weight_g':
                module_list[-1].weight_g.data.clamp_(min=0.0,max=value)
            elif key.split('.')[-1] == 'weight' and len(module_list[-1].weight.data.size()) > 1:
                module_list[-1].weight.data.renorm_(p=2,dim=0,maxnorm=value)

class BassetNormCat2(nn.Module):
    def __init__(self, dropout_prob=0.3, gene_drop_lvl=1):
        super(BassetNormCat2, self).__init__()
        self.conv1 = Conv1dNorm(4, 100, 19, stride=1, padding=9, weight_norm=False)
        self.conv1a = Conv1dNorm(4, 100, 11, stride=1, padding=5, weight_norm=False)
        self.conv1b = Conv1dNorm(4, 100, 25, stride=1, padding=12, weight_norm=False)
        self.conv2 = Conv1dNorm(300, 200, 11, stride=1, padding=0, weight_norm=False)
        self.conv3 = Conv1dNorm(200, 200, 7, stride=1, padding=0, weight_norm=False)
        self.maxpool_4 = nn.MaxPool1d(4,padding=0)
        self.maxpool_3 = nn.MaxPool1d(3,padding=0)
        self.genelinear = LinearNorm(19795, 500, weight_norm=False)
        self.linear1 = LinearNorm(200*13+500, 1000, weight_norm=False)
        self.linear2 = LinearNorm(1000, 1000, weight_norm=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(1000, 1)
        self.gdl = gene_drop_lvl
    def forward(self, x, geneexpr):
        out1 = F.relu(self.conv1(x)) # 3 of these
        out1a = F.relu(self.conv1a(x))
        out1b = F.relu(self.conv1b(x))
        out = self.maxpool_3(torch.cat([out1,out1a,out1b],dim=1)) # (?, 300, 600)
        out = F.pad(out,(5,5))
        out = F.relu(self.conv2(out)) # (?, 300, 140)
        out = self.maxpool_4(out) # (?, 300, 35)
        out = F.pad(out,(3,3))
        out = F.relu(self.conv3(out)) # (?, 500, 32)
        out = F.pad(out,(1,1))
        out = self.maxpool_4(out) # (?, 500, 8)
        out = out.view(-1, 200*13) # (?, 500*8)
        if self.gdl == 0:
            geneexpr = self.dropout(geneexpr)
            geneexpr = F.relu(self.genelinear(geneexpr))
        elif self.gdl == 1:
            geneexpr = F.relu(self.genelinear(geneexpr)) # (?, 500)
            geneexpr = self.dropout(geneexpr)
        out = torch.cat([out, geneexpr], dim=1) # (?, 200*13+500)
        out = F.relu(self.linear1(out)) # (?, 800)
        out = self.dropout(out)
        out = F.relu(self.linear2(out)) # (?, 800)
        out = self.dropout(out)
        return self.output(out) # (?, 1)
    def clip_norms(self, value):
        key_chain = [ key for key in self.state_dict().keys() if 'weight' in key ]
        for key in key_chain:
            module_list = [self]
            for key_level in key.split('.')[:-1]:
                module_list.append( getattr(module_list[-1], key_level) )
            if key.split('.')[-1] == 'weight_g':
                module_list[-1].weight_g.data.clamp_(min=0.0,max=value)
            elif key.split('.')[-1] == 'weight' and len(module_list[-1].weight.data.size()) > 1:
                module_list[-1].weight.data.renorm_(p=2,dim=0,maxnorm=value)
                
class BassetNormCat_bilin(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(BassetNormCat_bilin, self).__init__()
        self.conv1 = Conv1dNorm(4, 300, 19, stride=1, padding=0, weight_norm=False)
        self.conv2 = Conv1dNorm(300, 200, 11, stride=1, padding=0, weight_norm=False)
        self.conv3 = Conv1dNorm(200, 200, 7, stride=1, padding=0, weight_norm=False)
        self.maxpool_4 = nn.MaxPool1d(4,padding=0)
        self.maxpool_3 = nn.MaxPool1d(3,padding=0)
        self.bilin = nn.Bilinear(200*13,19795,200*13+500)
        self.linear1 = LinearNorm(200*13+500, 1000, weight_norm=False)
        self.linear2 = LinearNorm(1000, 1000, weight_norm=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(1000, 1)
    def forward(self, x, geneexpr):
        #if sparse_in: # (?, 600, 4)
        #    in_seq = to_one_hot(x, n_dims=4).permute(0,3,1,2).squeeze()
        #else:
        #    in_seq = x.squeeze()
        x = F.pad(x,(9,9))
        out = F.relu(self.conv1(x)) # (?, 4, 580)
        out = self.maxpool_3(out) # (?, 30, 145)
        out = F.pad(out,(5,5))
        out = F.relu(self.conv2(out)) # (?, 300, 140)
        out = self.maxpool_4(out) # (?, 300, 35)
        out = F.pad(out,(3,3))
        out = F.relu(self.conv3(out)) # (?, 500, 32)
        out = F.pad(out,(1,1))
        out = self.maxpool_4(out) # (?, 500, 8)
        out = out.view(-1, 200*13) # (?, 500*8)
        out = self.bilin(out,geneexpr) # (?,200*13+500)
        out = F.relu(self.linear1(out)) # (?, 800)
        out = self.dropout(out)
        out = F.relu(self.linear2(out)) # (?, 800)
        out = self.dropout(out)
        return self.output(out) # (?, 1)
    def clip_norms(self, value):
        key_chain = [ key for key in self.state_dict().keys() if 'weight' in key ]
        for key in key_chain:
            module_list = [self]
            for key_level in key.split('.')[:-1]:
                module_list.append( getattr(module_list[-1], key_level) )
            if key.split('.')[-1] == 'weight_g':
                module_list[-1].weight_g.data.clamp_(min=0.0,max=value)
            elif key.split('.')[-1] == 'weight' and len(module_list[-1].weight.data.size()) > 1:
                module_list[-1].weight.data.renorm_(p=2,dim=0,maxnorm=value)