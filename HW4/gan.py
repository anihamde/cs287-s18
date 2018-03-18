import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.distributions import Normal
import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser.add_argument('--latent_dim','-ld',type=int,default=32,help='Latent dimension')
parser.add_argument('--batch_size','-bs',type=int,default=100,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=50,help='Number of epochs')
parser.add_argument('--learning_rate','-lr',type=float,default=0.01,help='Learning rate')
args = parser.parse_args()

LATENT_DIM = args.latent_dim
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
learning_rate = args.learning_rate

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
train_img = torch.stack([d[0] for d in train_dataset])
train_label = torch.LongTensor([d[1] for d in train_dataset])
test_img = torch.stack([d[0] for d in test_dataset])
test_label = torch.LongTensor([d[1] for d in test_dataset])
# print(train_img[0])
print(train_img.size(), train_label.size(), test_img.size(), test_label.size())

# MNIST does not have an official train dataset. So we will use the last 10000 training points as your validation set.
val_img = train_img[-10000:].clone()
val_label = train_label[-10000:].clone()
train_img = train_img[:-10000]
train_label = train_label[:-10000]

train = torch.utils.data.TensorDataset(train_img, train_label)
val = torch.utils.data.TensorDataset(val_img, val_label)
test = torch.utils.data.TensorDataset(test_img, test_label)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

img, label = next(iter(train_loader))
print(img.size(),label.size())



# TODO: wait is the discriminator trying to tell which number it is, or 
# just tell whether it's real or fake? if the former, how do I do my loss?

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(LATENT_DIM, 100)
        self.linear2 = nn.Linear(100, 784)
    def forward(self, z):
        out = self.linear2(F.relu(self.linear1(z)))
        return out.view(-1,28,28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(784, 100)
        self.linear2 = nn.Linear(100, 1)
    def forward(self, x):
        x = x.view(-1,784)
        return F.sigmoid(self.linear2(F.relu(self.linear1(x))))

G = Generator()
D = Discriminator()
G.cuda()
D.cuda()
optim_gen = torch.optim.SGD(G.parameters(), lr=learning_rate)
optim_disc = torch.optim.SGD(D.parameters(), lr=learning_rate)
seed_distribution = Normal(V(torch.zeros(BATCH_SIZE, LATENT_DIM)), 
                           V(torch.ones(BATCH_SIZE, LATENT_DIM)))

for epoch in range(NUM_EPOCHS):
    total_gen_loss = 0
    total_disc_loss = 0
    total = 0
    for img, label in train_loader:
        if img.size(0) < BATCH_SIZE: continue
        img = img.squeeze(1) # there's an extra dimension for some reason
        img = img.cuda()
        # Grad discriminator real: -E[log(D(x))]
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        d = D(img)
        loss_a = 0.5 * -d.log().mean()
        loss_a.backward()
        # Grad discriminator fake: -E[log(1 - D(G(z)) )]
        seed = seed_distribution.sample()
        x_fake = G(seed)
        d = D(x_fake.detach())
        loss_b = 0.5 * -(1 - d + 1e-10).log().mean()
        loss_b.backward()
        optim_disc.step()
        total_disc_loss += loss_a.data[0] + loss_b.data[0]
        # Grad generator: E[log(1 - D(G(z)))]
        optim_disc.zero_grad()
        d = D(x_fake) # no detach here
        loss_c = (1 - d + 1e-10).log().mean()
        # loss_c = -(d + 1e-10).log().mean()
        loss_c.backward()
        optim_gen.step()
        total_gen_loss += loss_c.data[0]
        total += 1
    print(i, total_disc_loss /  total, total_gen_loss / total)

################### VISUALIZATION ########################
# section has code to generate a bunch and plot discriminator's results

# viz 1: generate a digit
seed_distribution = Normal(V(torch.zeros(1,LATENT_DIM)), 
                        V(torch.ones(1,LATENT_DIM)))
def graph():
    seed = seed_distribution.sample()
    x = G(seed) # 1,28,28
    plt.imshow(x[0].data.numpy())

# viz 2: interpolate
z1 = seed_distribution.sample()
z2 = seed_distribution.sample()
all = []
for k in np.arange(0,1.1,0.2):
    z = k * z1 + (1 - k) * z2
    x = G(z)
    all.append(x[0].data.numpy())
    plt.imshow(x[0].data.numpy())