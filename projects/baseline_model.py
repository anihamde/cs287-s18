import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
    def __init__(self, dropout_prob=0.5):
        super(Classic, self).__init__()
        self.conv1 = conv(4, 300, 21, stride=1, pad=0, bn=True)
        self.conv2 = conv(300, 300, 6, stride=1, pad=0, bn=True)
        self.conv3 = conv(300, 500, 4, stride=1, pad=0, bn=True)
        self.maxpool= nn.MaxPool1d(4,padding=0)
        self.linear = nn.Linear(500*8, 800)
        self.dropout= nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(800, 164)
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
    def __init__(self, dropout_prob=0.3):
        super(BassetNorm, self).__init__()
        self.conv1 = Conv1dNorm(4, 300, 19, stride=1, padding=0, weight_norm=False)
        self.conv2 = Conv1dNorm(300, 200, 11, stride=1, padding=0, weight_norm=False)
        self.conv3 = Conv1dNorm(200, 200, 7, stride=1, padding=0, weight_norm=False)
        self.maxpool_4 = nn.MaxPool1d(4,padding=0)
        self.maxpool_3 = nn.MaxPool1d(3,padding=0)
        self.linear1 = LinearNorm(200*13, 1000, weight_norm=False)
        self.linear2 = LinearNorm(1000, 1000, weight_norm=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(1000, 164)
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
            elif key.split('.')[-1] == 'weight':
                #module_list[-1] = nn.utils.weight_norm(module_list[-1])
                #module_list[-1].weight_g.data.clamp_(min=0.0,max=value)
                #torch.nn.utils.remove_weight_norm(module_list[-1])
                print(module_list[-1])
                print(module_list[-1].weight.data.size())
                module_list[-1].weight.data.renorm_(p=2,dim=0,maxnorm=value)
    
class Basset(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(Basset, self).__init__()
        self.conv1 = conv(4, 300, 19, stride=1, pad=0, bn=True)
        self.conv2 = conv(300, 200, 11, stride=1, pad=0, bn=True)
        self.conv3 = conv(200, 200, 7, stride=1, pad=0, bn=True)
        self.maxpool_4 = nn.MaxPool1d(4,padding=0)
        self.maxpool_3 = nn.MaxPool1d(3,padding=0)
        self.linear1 = nn.Linear(200*10, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(1000, 164)
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
    def __init__(self, dropout_prob=0.5):
        super(DeepSEA, self).__init__()
        self.conv1 = conv(4, 320, 8, stride=1, pad=2, bn=True)
        self.conv2 = conv(320, 480, 8, stride=1, pad=1, bn=True)
        self.conv3 = conv(480, 960, 8, stride=1, pad=0, bn=True)
        self.maxpool= nn.MaxPool1d(4,padding=0)
        self.linear = nn.Linear(960*29, 925)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_4 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.output = nn.Linear(925, 164)
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
