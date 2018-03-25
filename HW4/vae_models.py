import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
import numpy as np

############## OLD WAY ################
class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 200)
        self.linear2 = nn.Linear(200, latent_dim)
        self.linear3 = nn.Linear(200, latent_dim)
    def forward(self, x):
        x = x.view(-1,784)
        h = F.relu(self.linear1(x))
        return self.linear2(h), self.linear3(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 200)
        self.linear2 = nn.Linear(200, 784)
    def forward(self, z):
        out = self.linear2(F.relu(self.linear1(z)))
        return out.view(-1,1,28,28)


############## DCGAN pytorch tutorial #################
def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)    

class Decoder1(nn.Module):
    def __init__(self, latent_dim=2, image_size=28, conv_dim=32):
        super(Decoder, self).__init__()
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

class Encoder1(nn.Module):
    def __init__(self, latent_dim=2, image_size=28, conv_dim=32):
        super(Encoder, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        # self.fc = conv(conv_dim*8, 1, int(image_size/16), 1, 0, False)
        self.linear1 = nn.Linear(conv_dim*8, latent_dim)
        self.linear2 = nn.Linear(conv_dim*8, latent_dim)
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05) # (?, 32, 14, 14)
        out = F.leaky_relu(self.conv2(out), 0.05) # (?, 64, 7, 7)
        out = F.leaky_relu(self.conv3(out), 0.05) # (?, 128, 3, 3)
        out = F.leaky_relu(self.conv4(out), 0.05) # (?, 256, 1, 1)
        out = out.squeeze()
        mean, logvar = self.linear1(out), self.linear2(out)
        return mean, logvar


############## CNN pytorch tutorial #################
# but the tutorial doesn't have an inverse-CNN, so this is totally BS!!
class Decoder2(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 32, 7, stride=1, padding=0),
            nn.BatchNorm2d(32))
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU())
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        print(x.size())
        out = self.fc(x)
        print(out.size())
        out = self.layer1(out)
        print(out.size())
        out = self.layer2(out)
        print(out.size())
        return out

# CNN pytorch tutorial
class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.linear1 = nn.Linear(7*7*32, latent_dim)
        self.linear2 = nn.Linear(7*7*32, latent_dim)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        mu, logvar = self.linear1(out), self.linear2(out)
        return mu, logvar

# VAE using reparameterization "rsample"
class NormalVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(NormalVAE, self).__init__()
        # Parameters phi and computes variational parameters lambda
        self.encoder = encoder
        # Parameters theta, p(x|z)
        self.decoder = decoder
    def forward(self, x_src):
        # Example variational parameters lambda
        mu, logvar = self.encoder(x_src)
        q_normal = Normal(loc=mu, scale=logvar.mul(0.5).exp()) # TODO: why are we multiplying by half?
        # Reparameterized sample.
        z_sample = q_normal.rsample()
        # z_sample = mu (no sampling)
        return self.decoder(z_sample), q_normal