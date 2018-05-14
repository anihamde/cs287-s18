import torch
import time
import subprocess
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import numpy as np
import argparse
import time
import pandas as pd
from helpers import timeSince, asMinutes, calc_auc
from vi_model import *
# import h5py
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from baseline_model import *
# import torch.distributions.constraints as constraints
# import pyro
# from pyro.optim import Adam
# from pyro.infer import SVI, Trace_ELBO
# import pyro.distributions as dist
# from __future__ import print_function
# import sys
# import os
# import math

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--model_type','-m',type=int,default=0,help='Model type')
parser.add_argument('--latent_dim','-ld',type=int,default=2,help='Latent dimension')
parser.add_argument('--batch_size','-bs',type=int,default=57,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=20,help='Number of epochs')
parser.add_argument('--learning_rate','-lr',type=float,default=0.0001,help='Learning rate')
parser.add_argument('--alpha','-a',type=float,default=1.0,help='Alpha (weight of KL term in Elbo)')
parser.add_argument('--model_file','-mf',type=str,default='stupidvae.pkl',help='Save model filename')
parser.add_argument('--init_stdev','-sd',type=float,default=0.0001,help='Weight init stdev')
args = parser.parse_args()

expn_pth = '~/57epigenomes.RPKM.pc'
# expn_pth = '/n/data_02/Basset/data/expn/roadmap/57epigenomes.RPKM.pc'
print("Reading gene expression data from:\n{}".format(expn_pth))
# Gene expression dataset
expn = pd.read_table(expn_pth,header=0)
col_names = expn.columns.values[1:]
expn = expn.drop(col_names[-1],axis=1) # 19795*57 right now
expn.columns = col_names
pinned_lookup = torch.nn.Embedding.from_pretrained(torch.FloatTensor(expn.as_matrix().T),freeze=True)
pinned_lookup.cuda()

torch.manual_seed(3435)
imgs = torch.poisson(pinned_lookup.weight)
# imgs = pinned_lookup.weight.round()
# imgs = pinned_lookup.weight
# train_img = train_img[:-10000].clone()
dat = torch.utils.data.TensorDataset(imgs, torch.zeros(57,1)) # placeholder arg required pytorch <0.4.0...
loader = torch.utils.data.DataLoader(dat, batch_size=args.batch_size, shuffle=True)
img, _ = next(iter(loader))
print(img.size())

encoder = Encoder(latent_dim=args.latent_dim)
esd = encoder.state_dict()
for param in esd:
    esd[param].data = torch.randn(esd[param].size())*args.init_stdev

encoder.load_state_dict(esd)

decoder = Decoder(latent_dim=args.latent_dim)
vae = NormalVAE(encoder, decoder)
vae.cuda()
# vae.load_state_dict(torch.load(args.model_file))

criterion = nn.PoissonNLLLoss(log_input=True)
optim = torch.optim.SGD(vae.parameters(), lr = args.learning_rate)
# p(z)
p = Normal(Variable(torch.zeros(args.batch_size, args.latent_dim)).cuda(), 
           Variable(torch.ones(args.batch_size, args.latent_dim)).cuda())

# TODO: to make this stochastic, shuffle and make smaller batches.
start = time.time()
vae.train() # TODO: switch btwn train and eval for appropriate parts
for epoch in range(args.num_epochs):
    # Keep track of reconstruction loss and total kl
    total_recon_loss = 0
    total_kl = 0
    total = 0
    for img, _ in loader:
        # no need to Variable(img).cuda()
        vae.zero_grad()
        out, q = vae(img) # out is decoded distro sample, q is distro
        kl = kl_divergence(q, p).sum() # KL term
        recon_loss = criterion(out, img) # reconstruction term # TODO: test
        loss = (recon_loss + args.alpha * kl) / args.batch_size
        total_recon_loss += recon_loss.item() / args.batch_size
        total_kl += kl.item() / args.batch_size
        total += 1
        loss.backward()
        optim.step()
    timenow = timeSince(start)
    print ('Time %s, Epoch [%d/%d], Recon Loss: %.4f, KL Loss: %.4f, ELBO Loss: %.4f' 
            %(timenow, epoch+1, args.num_epochs, total_recon_loss/total , total_kl/total, (total_recon_loss+total_kl)/total))
    # TODO: maybe print every 100 batches?
