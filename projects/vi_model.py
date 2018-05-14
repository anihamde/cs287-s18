import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
import numpy as np

############## MLP ################
class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(19795, 200)
        self.linear2 = nn.Linear(200, latent_dim)
        self.linear3 = nn.Linear(200, latent_dim)
    def forward(self, x):
        h = F.relu(self.linear1(x))
        return self.linear2(h), self.linear3(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 200)
        self.linear2 = nn.Linear(200, 19795)
    def forward(self, z):
        out = self.linear2(F.softplus(self.linear1(z))) # soft relu is better for counts
        return out

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
        q_normal = Normal(loc=mu, scale=logvar.mul(0.5).exp())
        # Reparameterized sample.
        z_sample = q_normal.rsample()
        # z_sample = mu (no sampling)
        return self.decoder(z_sample), q_normal