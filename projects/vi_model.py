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
        h = F.relu(self.linear1(x)) # TODO: pyro tries a softplus here?
        return self.linear2(h), self.linear3(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 200)
        self.linear2 = nn.Linear(200, 19795)
    def forward(self, z):
        out = self.linear2(F.softplus(self.linear1(z)))
        # soft relu is better for counts. pyro does a softplus here
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

# define a PyTorch module for the VAE, with pyro methods
# encoder parameterizes diagonal gaussian distribution q(z|x)
# decoder parameterizes observation likelihood p(x|z)
# They put a sigmoid cap on their decoder. should i put a log cap (link) on mine?
class PyroVAE(nn.Module):
    # our latent space is 50-dimensional (and we use prespecified hidden_size)
    def __init__(self, latent_dim=50, hidden_size=400, use_cuda=False):
        super(VAE, self).__init__()
        # fixed hidden dim for now
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.cuda()
        self.latent_dim = latent_dim

    # define the model p(x|z)p(z). here x is bs,19795. TODO: how does this translate to 56 case?
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.iarange("data", x.size(0)):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.size(0), self.latent_dim)))
            z_scale = x.new_ones(torch.Size((x.size(0), self.latent_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
            # decode the latent code z
            log_img = self.decoder.forward(z)
            # score against actual images
            pyro.sample("obs", dist.Poisson(log_img.exp()).independent(1), obs=x.reshape(-1, 19795))
            # return the loc so we can visualize it later
            return log_img.exp()

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.iarange("data", x.size(0)):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        log_img = self.decoder(z)
        return log_img.exp()

class PyroVI(nn.Module):
    # our latent space is 50-dimensional (and we use prespecified hidden_size)
    def __init__(self, latent_dim=50, hidden_size=400, use_cuda=False):
        super(VAE, self).__init__()
        # fixed hidden dim for now
        self.theta = Decoder(latent_dim)
        self.cuda()
        self.latent_dim = latent_dim

    # define the model p(x|z)p(z). here x is bs,19795.
    def model(self, x):
        # register PyTorch module `theta` with Pyro
        pyro.module("theta", self.theta)
        with pyro.iarange("data", x.size(0)):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.size(0), self.latent_dim)))
            z_scale = x.new_ones(torch.Size((x.size(0), self.latent_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
            # decode the latent code z
            log_img = self.decoder.forward(z)
            # score against actual images
            pyro.sample("obs", dist.Poisson(log_img.exp()).independent(1), obs=x.reshape(-1, 19795))
            # return the loc so we can visualize it later
            return log_img.exp()

    # are we subsampling?
    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        z_loc = pyro.param("z_loc", torch.randn(56,self.latent_dim))
        z_scale = pyro.param("z_scale", torch.randn(56,self.latent_dim).mul(0.5).exp())
        with pyro.iarange("data", x.size(0)):
            # use the encoder to get the parameters used to define q(z|x)
            # z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))