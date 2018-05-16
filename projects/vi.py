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

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--model_type','-m',type=int,default=0,help='Model type')
parser.add_argument('--latent_dim','-ld',type=int,default=2,help='Latent dimension')
parser.add_argument('--batch_size','-bs',type=int,default=56,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=20,help='Number of epochs')
parser.add_argument('--clip','-c',type=float,default=5.0,help='Max norm for weight clipping')
parser.add_argument('--learning_rate1','-lr1',type=float,default=0.0001,help='Learning rate for theta')
parser.add_argument('--learning_rate2','-lr2',type=float,default=0.0001,help='Learning rate for lambda')
parser.add_argument('--alpha','-a',type=float,default=1.0,help='Alpha (weight of KL term in Elbo)')
parser.add_argument('--model_file','-mf',type=str,default='stupidvi.pkl',help='Save model filename')
parser.add_argument('--init_stdev','-sd',type=float,default=0.001,help='Weight init stdev')
args = parser.parse_args()

expn_pth = '/n/data_02/Basset/data/expn/roadmap/57epigenomes.RPKM.pc'
print("Reading gene expression data from:\n{}".format(expn_pth))
# Gene expression dataset
expn = pd.read_table(expn_pth,header=0)
col_names = expn.columns.values[1:]
expn = expn.drop(col_names[-1],axis=1) # 19795*57 right now # TODO: is this all right?
expn.columns = col_names
pinned_lookup = torch.nn.Embedding.from_pretrained(torch.FloatTensor(expn.as_matrix().T[1:]),freeze=True) # [1:] is new!
pinned_lookup.cuda()

torch.manual_seed(3435)
imgs = torch.poisson(pinned_lookup.weight) # discretize data
# imgs = pinned_lookup.weight.round()
# imgs = pinned_lookup.weight
dat = torch.utils.data.TensorDataset(imgs, torch.zeros(56,1)) # placeholder arg required pytorch <0.4.0...
loader = torch.utils.data.DataLoader(dat, batch_size=args.batch_size, shuffle=False)
img, _ = next(iter(loader))
print(img.size())

theta = Decoder(latent_dim=args.latent_dim)
if True: # initialize weights with smaller stdev to prevent instability
    dsd = theta.state_dict()
    for param in dsd:
        dsd[param].data = torch.randn(dsd[param].size())*args.init_stdev
    theta.load_state_dict(dsd)

theta.cuda()
# theta.load_state_dict(torch.load(args.model_file))

criterion = nn.PoissonNLLLoss(log_input=True, size_average=False)
optim1 = torch.optim.SGD(theta.parameters(), lr = args.learning_rate1)
p = Normal(Variable(torch.zeros(args.batch_size, args.latent_dim)).cuda(), 
           Variable(torch.ones(args.batch_size, args.latent_dim)).cuda()) # p(z)
mu = Variable(torch.randn(args.batch_size,args.latent_dim).cuda(),requires_grad=True) # variational parameters
logvar = Variable(torch.randn(args.batch_size,args.latent_dim).cuda(),requires_grad=True)
optim2 = torch.optim.SGD([mu,logvar], lr = args.learning_rate2)

# TODO: to make this stochastic, shuffle and make smaller batches.
start = time.time()
theta.train()
for epoch in range(args.num_epochs*2):
    # Keep track of reconstruction loss and total kl
    total_recon_loss = 0
    total_kl = 0
    total = 0
    for img, _ in loader:
        # no need to Variable(img).cuda()
        optim1.zero_grad()
        optim2.zero_grad()
        q = Normal(loc=mu, scale=logvar.mul(0.5).exp())
        # Reparameterized sample.
        qsamp = q.rsample()
        kl = kl_divergence(q, p).sum() # KL term
        out = theta(qsamp)
        recon_loss = criterion(out, img) # reconstruction term
        loss = (recon_loss + args.alpha * kl) / args.batch_size
        total_recon_loss += recon_loss.item() / args.batch_size
        total_kl += kl.item() / args.batch_size
        total += 1
        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm(theta.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm(mu, args.clip)
            torch.nn.utils.clip_grad_norm(theta.parameters(), args.clip)
        if epoch % 2:
            optim1.step()
            wv = 'Theta'
            print(theta.linear1.weight[:56:4])
            print(theta.linear2.weight)
        else:
            optim2.step()
            wv = 'Lambda'
            print(mu[:56:4])
            print(logvar[:56:4])
    timenow = timeSince(start)
    print ('Time %s, Epoch [%d/%d], Tuning %s, Recon Loss: %.4f, KL Loss: %.4f, ELBO Loss: %.4f' 
            %(timenow, (epoch+2)//2, args.num_epochs, wv, total_recon_loss/total , total_kl/total, (total_recon_loss+total_kl)/total))
    # TODO: add eval loop for big VAE
    torch.save(theta.state_dict(), args.model_file)
