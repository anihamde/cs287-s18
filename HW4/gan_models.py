import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# this is almost exactly identical to vae_models.py
# the main difference is the latent_dim (32 vs 2) and the last discriminator layer

############## OLD WAY ################
class Generator(nn.Module):
    def __init__(self, latent_dim=32):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 100)
        self.linear2 = nn.Linear(100, 784)
    def forward(self, z):
        out = self.linear2(F.relu(self.linear1(z)))
        return out.view(-1,1,28,28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(784, 100)
        self.linear2 = nn.Linear(100, 1)
    def forward(self, x):
        x = x.view(-1,784)
        return F.sigmoid(self.linear2(F.relu(self.linear1(x))))


############## DCGAN pytorch tutorial #################
def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)    

class Generator(nn.Module):
    def __init__(self, latent_dim=32, image_size=28, conv_dim=32):
        super(Generator, self).__init__()
        self.fc = deconv(latent_dim, conv_dim*8, 2, stride=1, pad=0, bn=False)
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 3) # hacky to change kernel size
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 1, 4, bn=False)
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z) # (?, 256, 2, 2)
        out = F.leaky_relu(self.deconv1(out), 0.05) # (?, 128, 4, 4)
        out = F.leaky_relu(self.deconv2(out), 0.05) # (?, 64, 7, 7)
        out = F.leaky_relu(self.deconv3(out), 0.05) # (?, 32, 14, 14)
        out = F.tanh(self.deconv4(out)) # (?, 1, 28, 28)
        return out

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, image_size=28, conv_dim=32):
        super(Discriminator, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        # self.fc = conv(conv_dim*8, 1, int(image_size/16), 1, 0, False)
        self.linear = nn.Linear(conv_dim*8, 1)
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05) # (?, 32, 14, 14)
        out = F.leaky_relu(self.conv2(out), 0.05) # (?, 64, 7, 7)
        out = F.leaky_relu(self.conv3(out), 0.05) # (?, 128, 3, 3)
        out = F.leaky_relu(self.conv4(out), 0.05) # (?, 256, 1, 1)
        out = out.squeeze()
        return F.sigmoid(self.linear(out))