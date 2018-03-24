import torch
from torch.autograd import Variable as V
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.distributions import Normal
import numpy as np
import argparse
import time
from helpers import timeSince, conv, deconv

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--latent_dim','-ld',type=int,default=32,help='Latent dimension')
parser.add_argument('--batch_size','-bs',type=int,default=100,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=50,help='Number of epochs')
parser.add_argument('--learning_rate','-lr',type=float,default=0.01,help='Learning rate')
parser.add_argument('--gen_file','-gf',type=str,default='stupidg.pkl',help='Save gen filename')
parser.add_argument('--disc_file','-df',type=str,default='stupidd.pkl',help='Save disc filename')
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





class Generator(nn.Module):
    """Generator containing 7 deconvolutional layers."""
    def __init__(self, z_dim=LATENT_DIM, image_size=128, conv_dim=64):
        super(Generator, self).__init__()
        self.fc = deconv(z_dim, conv_dim*8, int(image_size/16), 1, 0, bn=False)
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)
        
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)      # If image_size is 64, output shape is as below.
        out = self.fc(z)                            # (?, 512, 4, 4)
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.deconv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.deconv3(out), 0.05)  # (?, 64, 32, 32)
        out = F.tanh(self.deconv4(out))             # (?, 3, 64, 64)
        return out

class Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""
    def __init__(self, image_size=128, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.fc = conv(conv_dim*8, 1, int(image_size/16), 1, 0, False)
        
    def forward(self, x):                         # If image_size is 64, output shape is as below.
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 32, 32)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 512, 4, 4)
        out = self.fc(out).squeeze()
        return out





# OLD GEN AND DISCRIM
# Discriminator just needs to distinguish real and fake digits

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.linear1 = nn.Linear(LATENT_DIM, 100)
#         self.linear2 = nn.Linear(100, 784)
#     def forward(self, z):
#         out = self.linear2(F.relu(self.linear1(z)))
#         return out.view(-1,28,28)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.linear1 = nn.Linear(784, 100)
#         self.linear2 = nn.Linear(100, 1)
#     def forward(self, x):
#         x = x.view(-1,784)
#         return F.sigmoid(self.linear2(F.relu(self.linear1(x))))

G = Generator()
D = Discriminator()
G.cuda()
D.cuda()
optim_gen = torch.optim.SGD(G.parameters(), lr=learning_rate)
optim_disc = torch.optim.SGD(D.parameters(), lr=learning_rate)
seed_distribution = Normal(V(torch.zeros(BATCH_SIZE, LATENT_DIM).cuda()), 
                           V(torch.ones(BATCH_SIZE, LATENT_DIM)).cuda())

start = time.time()
for epoch in range(NUM_EPOCHS):
    total_gen_loss = 0
    total_disc_loss = 0
    total = 0
    G.train()
    D.train()
    for img, label in train_loader:
        if img.size(0) < BATCH_SIZE: continue
        img = img.squeeze(1) # there's an extra dimension for some reason
        img = V(img).cuda()
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
        total_disc_loss += loss_a.item() + loss_b.item()
        # Grad generator: E[log(1 - D(G(z)))]
        optim_disc.zero_grad()
        d = D(x_fake) # no detach here
        loss_c = (1 - d + 1e-10).log().mean()
        # loss_c = -(d + 1e-10).log().mean()
        loss_c.backward()
        optim_gen.step()
        total_gen_loss += loss_c.item()
        total += 1
    timenow = timeSince(start)
    print ('Time %s, Epoch [%d/%d], D Loss: %.4f, G Loss: %.4f, Total Loss: %.4f' 
            %(timenow, epoch+1, NUM_EPOCHS, total_disc_loss/total, total_gen_loss/total, (total_disc_loss+total_gen_loss)/total))

torch.save(G.state_dict(), args.gen_file)
torch.save(D.state_dict(), args.disc_file)
