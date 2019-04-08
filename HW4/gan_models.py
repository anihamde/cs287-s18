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
        return torch.sigmoid(out.view(-1,1,28,28))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(784, 10000)
        self.linear2 = nn.Linear(10000, 1)
    def forward(self, x):
        x = x.view(-1,784)
        return torch.sigmoid(self.linear2(F.relu(self.linear1(x))))


############## DCGAN pytorch tutorial #################
def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)    

# 664933 params
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
        out = torch.sigmoid(self.deconv4(out)) # (?, 1, 28, 28)
        return out

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

# 690273 params
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
        return torch.sigmoid(self.linear(out))

# 664933 params
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
        out = torch.sigmoid(self.deconv4(out)) # (?, 1, 28, 28)
        return out

#2394849 params
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
        return torch.sigmoid(self.output(out).squeeze())

# 817074 params
class Generator3(nn.Module):
    def __init__(self, latent_dim=32):
        super(Generator3, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 10000)
        self.linear2 = nn.Linear(10000, 784)
    def forward(self, z):
        out = self.linear2(F.relu(self.linear1(z)))
        return torch.sigmoid(out.view(-1,1,28,28))

# 2394849 params
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
        return torch.sigmoid(self.output(out).squeeze())

# 664993 params
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
        out = torch.sigmoid(self.deconv4(out)) # (?, 1, 28, 28)
        return out

# 7860001 params
class Discriminator4(nn.Module):
    def __init__(self):
        super(Discriminator4, self).__init__()
        self.linear1 = nn.Linear(784, 10000)
        self.linear2 = nn.Linear(10000, 1)
    def forward(self, x):
        x = x.view(-1,784)
        return torch.sigmoid(self.linear2(F.relu(self.linear1(x))))


# AM221 model
# 4517891 params
class Generator5(nn.Module):
    def __init__(self, d=64):
        super(Generator5, self).__init__()
        self.d = d
        self.linear = nn.Linear(100, 2*2*d*8)
        self.linear_bn = nn.BatchNorm1d(2*2*d*8)
        self.deconv1 = nn.ConvTranspose2d(d*8, d*4, 5, 2, 1) # changed things
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 5, 2, 2)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 5, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 5, 2, 1) # hey, 3 or 1 here?
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self, input):
        x = F.relu(self.linear_bn(self.linear(input)))
        x = x.view(-1, self.d*8, 2, 2)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = x[:,:,:-1,:-1] # hacky way to get shapes right (like "SAME" in tf)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = x[:,:,:-1,:-1]
        x = torch.tanh(self.deconv4(x))
        x = x[:,:,:-1,:-1]
        return x

# 4310401 params
class Discriminator5(nn.Module):
    def __init__(self, d=64):
        super(Discriminator5, self).__init__()
        self.d = d
        self.conv1 = nn.Conv2d(1, d, 5, 2, 2) # hey, 3 or 1 here?
        self.conv2 = nn.Conv2d(d, d*2, 5, 2, 2)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 5, 2, 2)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 5, 2, 2)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.linear = nn.Linear(2*2*d*8, 1)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = x.view(-1, 2*2*self.d*8)
        x = torch.sigmoid(self.linear(x))
        return x

# (CS287 gans all run in 20 seconds ish, mine (model 5) takes 5 minutes. 
# Backprop eats up most time. It’s much bigger so this makes sense. Can I go faster by increasing batch size?)