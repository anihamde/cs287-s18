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
        self.linear1 = nn.Linear(latent_dim, 10000)
        self.linear2 = nn.Linear(10000, 784)
    def forward(self, z):
        out = self.linear2(F.relu(self.linear1(z)))
        return F.sigmoid(out.view(-1,1,28,28))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(784, 10000)
        self.linear2 = nn.Linear(10000, 1)
    def forward(self, x):
        x = x.view(-1,784)
        return F.sigmoid(self.linear2(F.relu(self.linear1(x))))


############## DCGAN pytorch tutorial #################
def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)    

class Generator1(nn.Module):
    def __init__(self, latent_dim=32, image_size=28, conv_dim=32):
        super(Generator1, self).__init__()
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
        out = F.sigmoid(self.deconv4(out)) # (?, 1, 28, 28)
        return out

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class Discriminator1(nn.Module):
    def __init__(self, image_size=28, conv_dim=32):
        super(Discriminator1, self).__init__()
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

class Generator2(nn.Module):
    def __init__(self, latent_dim=32, image_size=28, conv_dim=32):
        super(Generator2, self).__init__()
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
        out = F.sigmoid(self.deconv4(out)) # (?, 1, 28, 28)
        return out

class Discriminator2(nn.Module):
    def __init__(self, image_size=28, conv_dim=32):
        super(Discriminator2, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(1, conv_dim, 4, stride=1, pad=1, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4, stride=1, pad=0)
        self.maxpool= nn.MaxPool2d(2,padding=1)
        self.linear = nn.Linear(conv_dim*2*6*6, 1024)
        self.output = nn.Linear(1024, 1)
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05) # (?, 32, 27, 27)
        out = self.maxpool(out) # (?, 32, 13, 13)
        out = F.leaky_relu(self.conv2(out), 0.05) # (?, 64, 10, 10)
        out = self.maxpool(out) # (?, 64, 6, 6)
        out = out.view(-1, self.conv_dim*2*6*6) # (?, 64*8*8)
        out = F.leaky_relu(self.linear(out), 0.05) # (?, 1024)
        return F.sigmoid(self.output(out).squeeze())

class Generator3(nn.Module):
    def __init__(self, latent_dim=32):
        super(Generator3, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 10000)
        self.linear2 = nn.Linear(10000, 784)
    def forward(self, z):
        out = self.linear2(F.relu(self.linear1(z)))
        return F.sigmoid(out.view(-1,1,28,28))

class Discriminator3(nn.Module):
    def __init__(self, image_size=28, conv_dim=32):
        super(Discriminator3, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(1, conv_dim, 4, stride=1, pad=1, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4, stride=1, pad=0)
        self.maxpool= nn.MaxPool2d(2,padding=1)
        self.linear = nn.Linear(conv_dim*2*6*6, 1024)
        self.output = nn.Linear(1024, 1)
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05) # (?, 32, 27, 27)
        out = self.maxpool(out) # (?, 32, 13, 13)
        out = F.leaky_relu(self.conv2(out), 0.05) # (?, 64, 10, 10)
        out = self.maxpool(out) # (?, 64, 6, 6)
        out = out.view(-1, self.conv_dim*2*6*6) # (?, 64*8*8)
        out = F.leaky_relu(self.linear(out), 0.05) # (?, 1024)
        return F.sigmoid(self.output(out).squeeze())

class Generator4(nn.Module):
    def __init__(self, latent_dim=32, image_size=28, conv_dim=32):
        super(Generator4, self).__init__()
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
        out = F.sigmoid(self.deconv4(out)) # (?, 1, 28, 28)
        return out

class Discriminator4(nn.Module):
    def __init__(self):
        super(Discriminator4, self).__init__()
        self.linear1 = nn.Linear(784, 10000)
        self.linear2 = nn.Linear(10000, 1)
    def forward(self, x):
        x = x.view(-1,784)
        return F.sigmoid(self.linear2(F.relu(self.linear1(x))))
