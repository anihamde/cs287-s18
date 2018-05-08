import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import h5py
from math import floor, ceil

def conv(c_in, c_out, k_size, stride=1, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv1d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm1d(c_out))
    return nn.Sequential(*layers)

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

class RoadmapDataset(Dataset):
    """Roadmap project dataset"""
    def __init__(self, h5_data, pd_expn, cell_types, segment='train'):
        targ_cells = [ (i,str(x,'utf-8')) for i,x in enumerate( list(h5_data['target_labels'][:]) ) if str(x,'utf-8') in cell_types ]
        expn_cells = list(pd_expn.columns.values)
        self.expn_dex = np.array([ expn_cells.index(x[1]) for x in targ_cells ]).astype('uint8')
        self.seq  = torch.ByteTensor( h5_data['{}_in'.format(segment)][:] )
        self.targ = torch.ByteTensor( h5_data['{}_out'.format(segment)][:, [ i for i,x in targ_cells ] ] )
    def __len__(self):
        assert self.seq.size(0) == self.targ.size(0), "Dataset size missmatch. Inputs: {}. Targets: {}.".format(self.seq.shape[0], self.targ.shape[0])
        self.n_seq = self.targ.size(0)
        self.n_cell= self.targ.size(1)
        return self.targ.size(0) * self.targ.size(1)
    def __getitem__(self, idx):
        seq_idx = idx %  self.n_seq
        cell_idx= idx // self.n_seq
        in_seq = self.seq[seq_idx,:,:]
        out_targ = self.targ[seq_idx,cell_idx:cell_idx+1]
        gene_idx = np.array(self.expn_dex[cell_idx]).flatten()
        return in_seq, gene_idx, out_targ

class Conv1dNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 bias=True, batch_norm=True, weight_norm=True):
        super(Conv1dNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride, padding, dilation, groups, bias)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, 
                                           affine=True, track_running_stats=True)
        
    def forward(self, input):
        try:
            return self.bn_layer( self.conv( input ) )
        except AttributeError:
            return self.conv( input )
        
class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 batch_norm=True, weight_norm=True):
        super(LinearNorm, self).__init__()
        self.linear  = nn.Linear(in_features, out_features, bias=True)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, 
                                           affine=True, track_running_stats=True)
    def forward(self, input):
        try:
            return self.bn_layer( self.linear( input ) )
        except AttributeError:
            return self.linear( input )

def BNC1_maker(in_dict):
    out_dict = {}
    width = 600
    # Layer 1
    out_dict['C1_C'] = in_dict['C1_C']
    out_dict['C1_W'] = in_dict['C1_W']
    out_dict['C1_P'] = ( floor((in_dict['C1_W'] - 1)/2),ciel((in_dict['C1_W'] - 1)/2) )
    out_dict['P1_W'] = in_dict['P1_W']
    hold = (width % in_dict['P1_W'])
    out_dict['P1_P'] = ( floor(hold/2),ciel(hold/2) )
    width = (width + hold) / in_dict['P1_W']
    # Layer 2
    out_dict['C2_C'] = in_dict['C2_C']
    out_dict['C2_W'] = in_dict['C2_W']
    out_dict['C2_P'] = floor((in_dict['C2_W'] - 1)/2),ciel((in_dict['C2_W'] - 1)/2)
    out_dict['P2_W'] = in_dict['P2_W']
    hold = (width % in_dict['P2_W'])
    out_dict['P2_P'] = ( floor(hold/2),ciel(hold/2) )
    width = (width + hold) / in_dict['P2_W']
    # Layer 3
    out_dict['C3_C'] = in_dict['C3_C']
    out_dict['C3_W'] = in_dict['C3_W']
    out_dict['C3_P'] = floor((in_dict['C3_W'] - 1)/2),ciel((in_dict['C3_W'] - 1)/2)
    out_dict['P3_W'] = in_dict['P3_W']
    hold = (width % in_dict['P3_W'])
    out_dict['P3_P'] = ( floor(hold/2),ciel(hold/2) )
    width = (width + hold) / in_dict['P3_W']
    # Layer Gene
    out_dict['G0_C'] = in_dict['G0_C']
    out_dict['G0_D'] = in_dict['G0_D']
    # Layer 4
    out_dict['L1_I'] = (width * in_dict['C3_C']) + in_dict['G0_C']
    out_dict['L1_O'] = in_dict['L1_O']
    out_dict['L1_D'] = in_dict['L1_D']
    # Layer 5
    out_dict['L2_I'] = out_dict['L1_O']
    out_dict['L2_O'] = in_dict['L2_O']
    out_dict['L2_D'] = in_dict['L2_D']
    # Layer 6
    out_dict['L3_I'] = out_dict['L2_O']
    out_dict['L3_O'] = 1

class BassetNormCat(nn.Module):
    def __init__(self, dropout_prob=0.3, model_dict):
        super(BassetNormCat, self).__init__()
        md = model_dict
        self.md = md
        self.conv1 = Conv1dNorm(4, md['C1_C'], md['C1_W'], stride=1, padding=md['C1_P'], weight_norm=False)
        self.conv2 = Conv1dNorm(md['C1_C'], md['C2_C'], md['C2_W'], stride=1, padding=md['C2_P'], weight_norm=False)
        self.conv3 = Conv1dNorm(md['C2_C'], md['C3_C'], md['C3_W'], stride=1, padding=md['C3_P'], weight_norm=False)
        self.maxpool1 = nn.MaxPool1d(md['P1_W'],padding=md['P1_P'])
        self.maxpool2 = nn.MaxPool1d(md['P1_W'],padding=md['P1_P'])
        self.maxpool3 = nn.MaxPool1d(md['P1_W'],padding=md['P1_P'])
        self.linearg = LinearNorm(19795, md['G0_C'], weight_norm=False)
        self.dropoutg= nn.Dropout(p=md['G0_D'])
        self.linear1 = LinearNorm(out_dict['L1_I'], out_dict['L1_O'], weight_norm=False)
        self.linear2 = LinearNorm(out_dict['L2_I'], out_dict['L2_O'], weight_norm=False)
        self.dropout1 = nn.Dropout(p=md['L1_D'])
        self.dropout2 = nn.Dropout(p=md['L2_D'])
        self.output = nn.Linear(1000, 1)
    def forward(self, x, geneexpr):
        #if sparse_in: # (?, 600, 4)
        #    in_seq = to_one_hot(x, n_dims=4).permute(0,3,1,2).squeeze()
        #else:
        #    in_seq = x.squeeze()
        md = self.md
        out = F.relu(self.conv1(x)) # (?, 4, 580)
        out = self.maxpool1(out) # (?, 30, 145)
        out = F.relu(self.conv2(out)) # (?, 300, 140)
        out = self.maxpool2(out) # (?, 300, 35)
        out = F.relu(self.conv3(out)) # (?, 500, 32)
        out = self.maxpool3(out) # (?, 500, 8)
        out = out.view(-1, md['L1_I'] - md['G0_C']) # (?, 500*8)
        geneexpr = F.relu(self.genelinear(geneexpr)) # (?, 500)
        geneexpr = self.dropoutg(geneexpr)
        out = torch.cat([out, geneexpr], dim=1) # (?, 200*13+500)
        out = F.relu(self.linear1(out)) # (?, 800)
        out = self.dropout1(out)
        out = F.relu(self.linear2(out)) # (?, 800)
        out = self.dropout2(out)
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
    def __init__(self, dropout_prob=0.3):
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
        geneexpr = F.relu(self.genelinear(geneexpr)) # (?, 500)
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
