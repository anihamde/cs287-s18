import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
train_dataset = datasets.MNIST(root='./data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = datasets.MNIST(root='./data/',
                           train=False, 
                           transform=transforms.ToTensor())

print(len(train_dataset))
print(len(test_dataset))
train_dataset[0][0]

torch.manual_seed(3435)
train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
train_label = torch.LongTensor([d[1] for d in train_dataset])
test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
test_label = torch.LongTensor([d[1] for d in test_dataset])
print(train_img[0])
print(train_img.size(), train_label.size(), test_img.size(), test_label.size())

val_img = train_img[-10000:].clone()
val_label = train_label[-10000:].clone()
train_img = train_img[:10000]
train_label = train_label[:10000]

train = torch.utils.data.TensorDataset(train_img, train_label)
val = torch.utils.data.TensorDataset(val_img, val_label)
test = torch.utils.data.TensorDataset(test_img, test_label)
BATCH_SIZE = 100
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

for datum in train_loader:
    img, label = datum
    print(img.size(), label.size())
    break

import numpy as np
from torch.autograd import Variable as V
import torch.nn.functional as F
# New stuff.
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

LATENT_DIM = 8

# generate 8-dim mean and variance for z given 2-dim x
# Compute the variational parameters for q
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(2, 200)
        self.linear2 = nn.Linear(200, LATENT_DIM)
        self.linear3 = nn.Linear(200, LATENT_DIM)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        return self.linear2(h), self.linear3(h)

# generate 2-dim x given 8-dim z
# Implement the generative model p(x | z)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(LATENT_DIM, 200)
        self.linear2 = nn.Linear(200, 2)

    def forward(self, z):
        return self.linear2(F.relu(self.linear1(z)))

# VAE using reparameterization "rsample"

class NormalVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(NormalVAE, self).__init__()

        # Parameters phi and computes variational parameters lambda
        self.encoder = encoder

        # Parameters theta, p(x | z)
        self.decoder = decoder
    
    def forward(self, x_src):
        # Example variational parameters lambda
        mu, logvar = self.encoder(x_src)
        
        q_normal = Normal(loc=mu, scale=logvar.mul(0.5).exp()) # shouldn't it be .exp().mul(0.5)?

        # Reparameterized sample.
        z_sample = q_normal.rsample()
        #z_sample = mu
        return self.decoder(z_sample), q_normal


BATCH_SIZE = 32

mse_loss = nn.L1Loss(size_average=False)

# Problem setup.
encoder = Encoder()
decoder = Decoder()
vae = NormalVAE(encoder, decoder)

# SGD
learning_rate = 0.02
optim = torch.optim.SGD(vae.parameters(), lr = learning_rate)

NUM_EPOCHS = 50

# Get samples.
p = Normal(V(torch.zeros(BATCH_SIZE, LATENT_DIM)), 
           V(torch.ones(BATCH_SIZE, LATENT_DIM)))


for epoch in range(NUM_EPOCHS):
    # Keep track of reconstruction loss and total kl
    total_loss = 0
    total_kl = 0
    total = 0
    alpha = 1
    for i, t in enumerate(X, BATCH_SIZE):
        if X[i:i+BATCH_SIZE].shape[0] < BATCH_SIZE : continue

        # Standard setup. 
        vae.zero_grad()
        x = V(torch.FloatTensor(X[i: i+BATCH_SIZE] )) 

        # Run VAE. 
        out, q = vae(x) # decoded distro sample, and distro
        kl = kl_divergence(q, p).sum() # kl term

        # actual loss
        loss = mse_loss(out, x) + alpha * kl
        # why mse_loss? expectation goes away because sampling, log prob just becomes (x-mu)^2/2sigma
        # instead we have 
        loss = loss / BATCH_SIZE

        # record keeping.
        total_loss += mse_loss(out, x).data / BATCH_SIZE
        total_kl += kl.data / BATCH_SIZE
        total += 1
        loss.backward()
        optim.step()
    graph_vae()
    print(i, total_loss[0] / total , total_kl[0] / total)


seed_distribution = Normal(V(torch.zeros(BATCH_SIZE, LATENT_DIM)), 
                        V(torch.ones(BATCH_SIZE, LATENT_DIM)))
def graph_vae():
    fig, axs = plt.subplots(1,1)
    all = []
    all_out = []
    for k in range(500):
        seed =  seed_distribution.sample()
        x = decoder(seed[0:1] )
        all.append(x.data[0].numpy())
       
    all = np.array(all)
    axs.scatter(all[:, 0], all[:, 1])

graph_vae()