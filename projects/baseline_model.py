import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import h5py

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
        
        
class Classic(nn.Module):
    def __init__(self, dropout_prob=0.5, output_labels=164):
        super(Classic, self).__init__()
        self.conv1 = conv(4, 300, 21, stride=1, pad=0, bn=True)
        self.conv2 = conv(300, 300, 6, stride=1, pad=0, bn=True)
        self.conv3 = conv(300, 500, 4, stride=1, pad=0, bn=True)
        self.maxpool= nn.MaxPool1d(4,padding=0)
        self.linear = nn.Linear(500*8, 800)
        self.dropout= nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(800, output_labels)
    def forward(self, x, sparse_in=True):
        #if sparse_in: # (?, 600, 4)
        #    in_seq = to_one_hot(x, n_dims=4).permute(0,3,1,2).squeeze()
        #else:
        #    in_seq = x.squeeze()
        out = F.relu(self.conv1(x)) # (?, 4, 580)
        out = self.maxpool(out) # (?, 30, 145)
        out = F.relu(self.conv2(out)) # (?, 300, 140)
        out = self.maxpool(out) # (?, 300, 35)
        out = F.relu(self.conv3(out)) # (?, 500, 32)
        out = self.maxpool(out) # (?, 500, 8)
        out = out.view(-1, 500*8) # (?, 500*8)
        out = F.relu(self.linear(out)) # (?, 800)
        out = self.dropout(out)
        return self.output(out) # (?, 164)

class BassetNorm(nn.Module):
    def __init__(self, dropout_prob=0.3, output_labels=164):
        super(BassetNorm, self).__init__()
        self.conv1 = Conv1dNorm(4, 300, 19, stride=1, padding=0, weight_norm=False)
        self.conv2 = Conv1dNorm(300, 200, 11, stride=1, padding=0, weight_norm=False)
        self.conv3 = Conv1dNorm(200, 200, 7, stride=1, padding=0, weight_norm=False)
        self.maxpool_4 = nn.MaxPool1d(4,padding=0)
        self.maxpool_3 = nn.MaxPool1d(3,padding=0)
        self.linear1 = LinearNorm(200*13, 1000, weight_norm=False)
        self.linear2 = LinearNorm(1000, 1000, weight_norm=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(1000, output_labels)
    def forward(self, x):
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
        out = F.relu(self.linear1(out)) # (?, 800)
        out = self.dropout(out)
        out = F.relu(self.linear2(out)) # (?, 800)
        out = self.dropout(out)
        return self.output(out) # (?, 164)
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
                
class BassetNormCat(nn.Module):
    def __init__(self, dropout_prob=0.3, gene_drop_lvl=1):
        super(BassetNormCat, self).__init__()
        self.conv1 = Conv1dNorm(4, 300, 19, stride=1, padding=0, weight_norm=False)
        self.conv2 = Conv1dNorm(300, 200, 11, stride=1, padding=0, weight_norm=False)
        self.conv3 = Conv1dNorm(200, 200, 7, stride=1, padding=0, weight_norm=False)
        self.maxpool_4 = nn.MaxPool1d(4,padding=0)
        self.maxpool_3 = nn.MaxPool1d(3,padding=0)
        self.genelinear = LinearNorm(19795, 500, batch_norm=False, weight_norm=False)
        self.linear1 = LinearNorm(200*13, 1000, batch_norm=False, weight_norm=False)
        self.linear2 = LinearNorm(1000, 1000, batch_norm=False, weight_norm=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(1000+500, 1)
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
        out = F.relu(self.linear1(out)) # (?, 800)
        out = self.dropout(out)
        out = F.relu(self.linear2(out)) # (?, 800)
        out = self.dropout(out)
        if self.gdl == 0:
            #geneexpr = self.dropout(geneexpr)
            geneexpr = F.relu(self.genelinear(geneexpr))
        elif self.gdl == 1:
            geneexpr = F.relu(self.genelinear(geneexpr)) # (?, 500)
            geneexpr = self.dropout(geneexpr)
        elif self.gdl == 2:
            geneexpr = F.normalize(self.genelinear(geneexpr), p=2, dim=1)
        out = torch.cat([out, geneexpr], dim=1) # (?, 200*13+500)
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
        self.linear1 = LinearNorm(200*13, 1000, weight_norm=False)
        self.linear2 = LinearNorm(1000+500, 2000, weight_norm=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(2000, 1)
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
        out = F.relu(self.linear1(out)) # (?, 800)
        out = self.dropout(out)
        if self.gdl == 0:
            geneexpr = self.dropout(geneexpr)
            geneexpr = F.relu(self.genelinear(geneexpr))
        elif self.gdl == 1:
            geneexpr = F.relu(self.genelinear(geneexpr)) # (?, 500)
            geneexpr = self.dropout(geneexpr)
        out = torch.cat([out, geneexpr], dim=1) # (?, 200*13+500)
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

                
class Basset(nn.Module):
    def __init__(self, dropout_prob=0.3, output_labels=164):
        super(Basset, self).__init__()
        self.conv1 = conv(4, 300, 19, stride=1, pad=0, bn=True)
        self.conv2 = conv(300, 200, 11, stride=1, pad=0, bn=True)
        self.conv3 = conv(200, 200, 7, stride=1, pad=0, bn=True)
        self.maxpool_4 = nn.MaxPool1d(4,padding=0)
        self.maxpool_3 = nn.MaxPool1d(3,padding=0)
        self.linear1 = nn.Linear(200*10, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(1000, output_labels)
    def forward(self, x, sparse_in=True):
        #if sparse_in: # (?, 600, 4)
        #    in_seq = to_one_hot(x, n_dims=4).permute(0,3,1,2).squeeze()
        #else:
        #    in_seq = x.squeeze()
        out = F.relu(self.conv1(x)) # (?, 4, 580)
        out = self.maxpool_3(out) # (?, 30, 145)
        out = F.relu(self.conv2(out)) # (?, 300, 140)
        out = self.maxpool_4(out) # (?, 300, 35)
        out = F.relu(self.conv3(out)) # (?, 500, 32)
        out = self.maxpool_4(out) # (?, 500, 8)
        out = out.view(-1, 200*10) # (?, 500*8)
        out = F.relu(self.linear1(out)) # (?, 800)
        out = self.dropout(out)
        out = F.relu(self.linear2(out)) # (?, 800)
        out = self.dropout(out)
        return self.output(out) # (?, 164)

class DeepSEA(nn.Module):
    def __init__(self, dropout_prob=0.5, output_labels=164):
        super(DeepSEA, self).__init__()
        self.conv1 = conv(4, 320, 8, stride=1, pad=2, bn=True)
        self.conv2 = conv(320, 480, 8, stride=1, pad=1, bn=True)
        self.conv3 = conv(480, 960, 8, stride=1, pad=0, bn=True)
        self.maxpool= nn.MaxPool1d(4,padding=0)
        self.linear = nn.Linear(960*29, 925)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_4 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.output = nn.Linear(925, output_labels)
    def forward(self, x):
        #if sparse_in: # (?, 600, 4)
        #    in_seq = to_one_hot(x, n_dims=4).permute(0,3,1,2).squeeze()
        #else:
        #    in_seq = x.squeeze()
        out = F.relu(self.conv1(x)) # (?, 4, 580)
        out = self.maxpool(out) # (?, 30, 145)
        out = self.dropout_2(out) # dropout
        out = F.relu(self.conv2(out)) # (?, 300, 140)
        out = self.maxpool(out) # (?, 300, 35)
        out = self.dropout_4(out) # dropout
        out = F.relu(self.conv3(out)) # (?, 500, 32)
        out = out.view(-1, 960*29) # (?, 500*8)
        out = self.dropout_5(out) # dropout
        out = F.relu(self.linear(out)) # (?, 800)
        return self.output(out) # (?, 164)

class DanQ(nn.Module):
    def __init__(self, dropout_prob_02=0.2, dropout_prob_03=0.5, hidden_size=512, num_layers=1,
                 bidirectional=True, output_labels=164):
        # TODO: weight initialize unif[-.05,.05], bias 0
        super(DanQ, self).__init__()
        self.conv1 = nn.Conv1d(4, 1024, 30, stride=1, padding=0)
        
        
        conv_weights = self.conv1.weight
        
        print(conv_weights.size())

        JASPAR_motifs = list(np.load('JASPAR_CORE_2016_vertebrates.npy', encoding = 'latin1'))

        reverse_motifs = [JASPAR_motifs[19][::-1,::-1], JASPAR_motifs[97][::-1,::-1], JASPAR_motifs[98][::-1,::-1], JASPAR_motifs[99][::-1,::-1], JASPAR_motifs[100][::-1,::-1], JASPAR_motifs[101][::-1,::-1]]
        JASPAR_motifs = JASPAR_motifs + reverse_motifs
        print(len(JASPAR_motifs))
        print(len(JASPAR_motifs[0]))
        print(len(JASPAR_motifs[0][0]))

        for i in xrange(len(JASPAR_motifs)):
            m = JASPAR_motifs[i][::-1,:]
            w = len(m)
            #conv_weights[0][i,:,:,0] = 0
            #start = (30-w)/2
            start = np.random.randint(low=3, high=30-w+1-3)
            conv_weights[0][i,:,start:start+w,0] = m.T - 0.25
            #conv_weights[1][i] = -0.5
            conv_weights[1][i] = np.random.uniform(low=-1.0,high=0.0)

        conv_layer.set_weights(conv_weights)

        
        
        
        self.maxpool = nn.MaxPool1d(15,padding=0)
        self.lstm = nn.LSTM(input_size=1024,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
        # other relevant args: nonlinearity, dropout
        # lstm input shape: seq_len,bs,input_size
        # hidden shape: num_layers*num_directions,bs,hidden_size 
        # output shape: seq_len,bs,hidden_size*num_directions
        self.dropout_2 = nn.Dropout(p=dropout_prob_02)
        self.dropout_3 = nn.Dropout(p=dropout_prob_03)
        self.linear = nn.Linear(39*1024,925)
        self.output = nn.Linear(925, output_labels)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.directions = bidirectional + 1
    def initHidden(self,bs):
        return (Variable(torch.zeros(self.num_layers*self.directions,bs,self.hidden_size).cuda()), 
                Variable(torch.zeros(self.num_layers*self.directions,bs,self.hidden_size).cuda()))
    def forward(self, x):
        out = F.relu(self.conv1(x)) # (?, 1024, 571)
        out = F.pad(out,(14,0)) # (?, 1024, 585)
        out = self.maxpool(out) # (?, 1024, 39)
        out = self.dropout_2(out) # (?, 1024, 39)
        out = out.permute(2,0,1) # (39, ?, 1024)
        out,_ = self.lstm(out, self.initHidden(out.size(1))) # (39, ?, 1024)
        out = self.dropout_3(out) # (39, ?, 1024)
        out = out.transpose(1,0).reshape(-1,39*1024) # (/, 39*1024)
        out = F.relu(self.linear(out)) # (?, 925)
        return self.output(out) # (?, 164)

# class DanQCat(nn.Module):
#     def __init__(self, dropout_prob_02=0.2, dropout_prob_03=0.5, hidden_size=512, num_layers=1,
#                  bidirectional=True, output_labels=1):
#         # TODO: weight initialize unif[-.05,.05], bias 0
#         super(DanQCat, self).__init__()
#         self.conv1 = nn.Conv1d(4, 1024, 30, stride=1, padding=0)
#         self.maxpool = nn.MaxPool1d(15,padding=0)
#         self.lstm = nn.LSTM(input_size=1024,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
#         # other relevant args: nonlinearity, dropout
#         # lstm input shape: seq_len,bs,input_size
#         # hidden shape: num_layers*num_directions,bs,hidden_size
#         # output shape: seq_len,bs,hidden_size*num_directions
#         self.dropout_2 = nn.Dropout(p=dropout_prob_02)
#         self.dropout_3 = nn.Dropout(p=dropout_prob_03)

#         # NEW
#         self.genelinear = LinearNorm(19795, 500, weight_norm=False)
#         self.linear = nn.Linear(39*1024+500,925)
        
#         self.output = nn.Linear(925, output_labels)
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.directions = bidirectional + 1
#     def initHidden(self,bs):
#         return (Variable(torch.zeros(self.num_layers*self.directions,bs,self.hidden_size).cuda()), 
#                 Variable(torch.zeros(self.num_layers*self.directions,bs,self.hidden_size).cuda()))
#     def forward(self, x, geneexpr):
#         out = F.relu(self.conv1(x)) # (?, 1024, 571)
#         out = F.pad(out,(14,0)) # (?, 1024, 585)
#         out = self.maxpool(out) # (?, 1024, 39)
#         out = self.dropout_2(out) # (?, 1024, 39)
#         out = out.permute(2,0,1) # (39, ?, 1024)
#         out,_ = self.lstm(out, self.initHidden(out.size(1))) # (39, ?, 1024)
#         out = self.dropout_3(out) # (39, ?, 1024)
#         out = out.transpose(1,0).reshape(-1,39*1024) # (/, 39*1024)

#         # NEW
#         geneexpr = F.relu(self.genelinear(geneexpr)) # (?, 500)
#         out = torch.cat([out,geneexpr], dim = 1) # (?, 39*1024+500)

#         out = F.relu(self.linear(out)) # (?, 925)
#         return self.output(out) # (?, 1)

class DanQCat(nn.Module):
    def __init__(self, dropout_prob_02=0.2, dropout_prob_03=0.5, hidden_size=160, num_layers=1,
                 bidirectional=True, output_labels=1):
        # TODO: weight initialize unif[-.05,.05], bias 0
        super(DanQCat, self).__init__()
        self.conv1 = nn.Conv1d(4, 320, 26, stride=1, padding=0)
        self.maxpool = nn.MaxPool1d(13,padding=0)
        self.lstm = nn.LSTM(input_size=320,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
        # other relevant args: nonlinearity, dropout
        # lstm input shape: seq_len,bs,input_size
        # hidden shape: num_layers*num_directions,bs,hidden_size
        # output shape: seq_len,bs,hidden_size*num_directions
        self.dropout_2 = nn.Dropout(p=dropout_prob_02)
        self.dropout_3 = nn.Dropout(p=dropout_prob_03)

        # NEW
        self.genelinear = LinearNorm(19795, 500, weight_norm=False)
        self.linear = nn.Linear(45*320+500,925)
        
        self.dropout_4 = nn.Dropout(p=0.2)
        
        self.output = nn.Linear(925, output_labels)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.directions = bidirectional + 1
    def initHidden(self,bs):
        return (Variable(torch.zeros(self.num_layers*self.directions,bs,self.hidden_size).cuda()), 
                Variable(torch.zeros(self.num_layers*self.directions,bs,self.hidden_size).cuda()))
    def forward(self, x, geneexpr):
        out = F.relu(self.conv1(x)) # (?, 320, 571)
        out = F.pad(out,(14,0)) # (?, 320, 585)
        out = self.maxpool(out) # (?, 320, 45)
        out = self.dropout_2(out) # (?, 320, 45)
        out = out.permute(2,0,1) # (45, ?, 320)
        out,_ = self.lstm(out, self.initHidden(out.size(1))) # (45, ?, 320)
        out = self.dropout_3(out) # (45, ?, 320)
        out = out.transpose(1,0).reshape(-1,45*320) # (/, 45*320)

        # NEW
        geneexpr = F.relu(self.genelinear(geneexpr)) # (?, 500)
        geneexpr = self.dropout_4(geneexpr)
        out = torch.cat([out,geneexpr], dim = 1) # (?, 45*320+500)

        out = F.relu(self.linear(out)) # (?, 925)
        return self.output(out) # (?, 1)

class DanQCat_attn(nn.Module):
    def __init__(self, dropout_prob_02=0.2, dropout_prob_03=0.5, hidden_size=160, num_layers=1,
                 bidirectional=True, output_labels=1):
        # TODO: weight initialize unif[-.05,.05], bias 0
        super(DanQCat_attn, self).__init__()
        self.conv1 = nn.Conv1d(4, 320, 26, stride=1, padding=0)
        self.maxpool = nn.MaxPool1d(13,padding=0)
        self.lstm = nn.LSTM(input_size=320,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
        # other relevant args: nonlinearity, dropout
        # lstm input shape: seq_len,bs,input_size
        # hidden shape: num_layers*num_directions,bs,hidden_size
        # output shape: seq_len,bs,hidden_size*num_directions
        self.dropout_2 = nn.Dropout(p=dropout_prob_02)
        self.dropout_3 = nn.Dropout(p=dropout_prob_03)

        # NEW
        self.genelinear = LinearNorm(19795, 2*hidden_size, weight_norm=False)
        self.linear = nn.Linear(45*320,925)
        
        self.dropout_4 = nn.Dropout(p=0.2)
        
        self.output = nn.Linear(925, output_labels)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.directions = bidirectional + 1
    def initHidden(self,bs):
        return (Variable(torch.zeros(self.num_layers*self.directions,bs,self.hidden_size).cuda()), 
                Variable(torch.zeros(self.num_layers*self.directions,bs,self.hidden_size).cuda()))
    def forward(self, x, geneexpr):
        out = F.relu(self.conv1(x)) # (?, 320, 571)
        out = F.pad(out,(14,0)) # (?, 320, 585)
        out = self.maxpool(out) # (?, 320, 45)
        out = self.dropout_2(out) # (?, 320, 45)
        out = out.permute(2,0,1) # (45, ?, 320)
        
        geneexpr = F.relu(self.genelinear(geneexpr)) # (?,160)
        attn = torch.zeros(out.size(0))
        hn = torch.zeros(out.size(0),2,len(x),160)
        
        for i in range(out.size(0)):
            out[i],hid = self.lstm(out[i].unsqueeze(0), self.initHidden(out.size(1)))
            hn[i] = hid[0]
#             print(geneexpr.size(),out.size())
#             inter = torch.squeeze(torch.matmul(geneexpr.view(geneexpr.size(0),1,320),out[i].view(out[i].size(0),320,1)))
#             print(hn[i].size(),inter.size())
            print(geneexpr.unsqueeze(1).size(),hn[i].permute(1,0,2).size())
            attn[i] = torch.squeeze(torch.bmm(hn[i].permute(1,0,2).contiguous().view(800,1,320),geneexpr.unsqueeze(1))) # attn is (45,800), hn is (45,2,800,160)
        
        hn = hn.view(45,800,320)
        attn = attn.unsqueeze(2).expand(-1,-1,320)
        
        out = torch.sum(torch.mul(attn,hn),0) # (45, ?, 320)
            
#         out = self.dropout_3(out) # (45, ?, 320)
#         out = out.transpose(1,0).reshape(-1,45*320) # (/, 45*320)

        out = F.relu(self.linear(out)) # (?, 925)
        return self.output(out) # (?, 1)
