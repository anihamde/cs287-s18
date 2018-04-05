import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def conv(c_in, c_out, k_size, stride=1, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv1d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class Basset(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(Basset, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(4, 300, 21, stride=1, pad=0, bn=False)
        self.conv2 = conv(4, 300, 6, stride=1, pad=0, bn=False)
        self.conv3 = conv(4, 500, 4, stride=1, pad=0, bn=False)
        self.maxpool= nn.MaxPool1d(4,padding=0)
        self.linear = nn.Linear(500*8, 800)
        self.dropout= nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(800, 164)
    def forward(self, x):
        out = F.relu(self.conv1(x)) # (?, 4, 1, 580)
        out = self.maxpool(out) # (?, 300, 1, 145)
        out = F.relu(self.conv2(out)) # (?, 300, 1, 140)
        out = self.maxpool(out) # (?, 300, 1, 35)
        out = F.relu(self.conv3(out)) # (?, 500, 1, 32)
        out = self.maxpool(out) # (?, 500, 1, 8)
        out = out.view(-1, 500*8) # (?, 500*8)
        out = F.relu(self.linear(out)) # (?, 800)
        out = self.dropout(out)
        return F.sigmoid(self.output(out).squeeze()) # (?, 164)

