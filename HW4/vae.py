import torch
from torch.autograd import Variable as V
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import numpy as np
import argparse
import time
from helpers import timeSince

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--latent_dim','-ld',type=int,default=2,help='Latent dimension')
parser.add_argument('--batch_size','-bs',type=int,default=100,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=50,help='Number of epochs')
parser.add_argument('--learning_rate','-lr',type=float,default=0.02,help='Learning rate')
parser.add_argument('--alpha','-a',type=float,default=1.0,help='Alpha (weight of KL term in Elbo)')
args = parser.parse_args()

LATENT_DIM = args.latent_dim
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
learning_rate = args.learning_rate
alpha = args.alpha

train_dataset = datasets.MNIST(root='./data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = datasets.MNIST(root='./data/',
                           train=False, 
                           transform=transforms.ToTensor())

print(len(train_dataset))
print(len(test_dataset))
# train_dataset[0][0]

torch.manual_seed(3435)
train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
train_label = torch.LongTensor([d[1] for d in train_dataset])
test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
test_label = torch.LongTensor([d[1] for d in test_dataset])
# print(train_img[0])
print(train_img.size(), train_label.size(), test_img.size(), test_label.size())

# MNIST does not have an official train dataset. So we will use the last 10000 training points as your validation set.
val_img = train_img[-10000:].clone()
val_label = train_label[-10000:].clone()
train_img = train_img[:-10000] # TODO: this should be -10000 right?
train_label = train_label[:-10000]

train = torch.utils.data.TensorDataset(train_img, train_label)
val = torch.utils.data.TensorDataset(val_img, val_label)
test = torch.utils.data.TensorDataset(test_img, test_label)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

img, label = next(iter(train_loader))
print(img.size(),label.size())

# TODO: what's the relevance of Variable in torch 0.4?

# generate latent-dim mean and variance for z given 784-dim x
# Compute the variational parameters for q
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 200)
        self.linear2 = nn.Linear(200, LATENT_DIM)
        self.linear3 = nn.Linear(200, LATENT_DIM)
    def forward(self, x):
        x = x.view(-1,784)
        h = F.relu(self.linear1(x))
        return self.linear2(h), self.linear3(h)

# generate 784-dim x given latent-dim z
# Implement the generative model p(x|z)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(LATENT_DIM, 200)
        self.linear2 = nn.Linear(200, 784)
    def forward(self, z):
        out = self.linear2(F.relu(self.linear1(z)))
        return out.view(-1,28,28)

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

# Problem setup.
mse_loss = nn.L1Loss(size_average=False) 
# L1 loss corresponds to Laplace p(x|z), L2 loss corresponds to Gaussian
encoder = Encoder()
decoder = Decoder()
vae = NormalVAE(encoder, decoder)
# vae.load_state_dict(torch.load('../models/stupidvae.pkl'))
vae.cuda()
optim = torch.optim.SGD(vae.parameters(), lr = learning_rate)

# p(z)
p = Normal(V(torch.zeros(BATCH_SIZE, LATENT_DIM).cuda()), 
           V(torch.ones(BATCH_SIZE, LATENT_DIM)).cuda())

start = time.time()
for epoch in range(NUM_EPOCHS):
    # Keep track of reconstruction loss and total kl
    total_recon_loss = 0
    total_kl = 0
    total = 0
    vae.train()
    for img, label in train_loader:
        if img.size(0) < BATCH_SIZE: continue
        img = img.squeeze(1) # there's an extra dimension for some reason
        img = V(img).cuda()
        vae.zero_grad()
        out, q = vae(img) # out is decoded distro sample, q is distro
        kl = kl_divergence(q, p).sum() # KL term
        recon_loss = mse_loss(out, img) # reconstruction term
        loss = (recon_loss + alpha * kl) / BATCH_SIZE
        total_recon_loss += recon_loss.item() / BATCH_SIZE
        total_kl += kl.item() / BATCH_SIZE
        total += 1
        loss.backward()
        optim.step()
    timenow = timeSince(start)
    print ('Time %s, Epoch [%d/%d], Recon Loss: %.4f, KL Loss: %.4f, ELBO Loss: %.4f' 
            %(timenow, epoch+1, NUM_EPOCHS, total_recon_loss/total , total_kl/total, (total_recon_loss+total_kl)/total))
    # TODO: maybe print every 100 batches?
    # TODO: add a val loop for early stopping (and for GAN too!)

# temporary code because matplotlib doesn't work
torch.save(vae.state_dict(), 'stupidvae.pkl')