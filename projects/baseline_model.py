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
