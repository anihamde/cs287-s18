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
from helpers import timeSince
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gan_models import *

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--model_type','-m',type=int,default=0,help='Model type')
parser.add_argument('--latent_dim','-ld',type=int,default=32,help='Latent dimension')
parser.add_argument('--batch_size','-bs',type=int,default=100,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=50,help='Number of epochs')
parser.add_argument('--learning_rate','-lr',type=float,default=0.0001,help='Learning rate')
parser.add_argument('--gen_file','-gf',type=str,default='stupidg.pkl',help='Save gen filename')
parser.add_argument('--disc_file','-df',type=str,default='stupidd.pkl',help='Save disc filename')
parser.add_argument('--track_space','-ts',action='store_true',help='Save 2D latent space viz, if ld=2')
args = parser.parse_args()

LATENT_DIM = args.latent_dim
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
learning_rate = args.learning_rate

# based on AM221 code, could've just loaded this into the DataLoader without the in-between steps
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


if args.model_type == 0:
    G = Generator(latent_dim = LATENT_DIM)
    D = Discriminator()
elif args.model_type == 1:
    G = Generator1(latent_dim = LATENT_DIM)
    D = Discriminator1()
elif args.model_type == 2:
    G = Generator2(latent_dim = LATENT_DIM)
    D = Discriminator2()
elif args.model_type == 3:
    G = Generator3(latent_dim = LATENT_DIM)
    D = Discriminator3()
elif args.model_type == 4:
    G = Generator4(latent_dim = LATENT_DIM)
    D = Discriminator4()
elif args.model_type == 5:
    print('did you make sure Latent dim is 100? else errors are coming!')
    G = Generator5()
    D = Discriminator5()

G.cuda()
D.cuda()
optim_gen = torch.optim.Adam(G.parameters(), lr=learning_rate)
optim_disc = torch.optim.SGD(D.parameters(), lr=learning_rate)
seed_distribution = Normal(V(torch.zeros(BATCH_SIZE, LATENT_DIM).cuda()), 
                           V(torch.ones(BATCH_SIZE, LATENT_DIM)).cuda())

start = time.time()
G.train() # TODO: switch between train and eval for appropriate parts
D.train()
for epoch in range(NUM_EPOCHS):
    total_gen_loss = 0
    total_disc_loss = 0
    total = 0
    for img, label in train_loader:
        if img.size(0) < BATCH_SIZE: continue
        img = V(img).cuda()
        # Grad discriminator real: -E[log(D(x))]
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        d = D(img)
        loss_a = -d.log().mean()
        loss_a.backward()
        # Grad discriminator fake: -E[log(1 - D(G(z)) )]
        seed = seed_distribution.sample()
        x_fake = G(seed)
        d = D(x_fake.detach())
        loss_b = -(1 - d + 1e-10).log().mean()
        loss_b.backward()
        optim_disc.step()
        total_disc_loss += loss_a.item() + loss_b.item()
        # Grad generator: E[log(1 - D(G(z)))]
        # optim_disc.zero_grad()
        d = D(x_fake) # no detach here
        # loss_c = (1 - d + 1e-10).log().mean()
        loss_c = -(d + 1e-10).log().mean()
        loss_c.backward()
        optim_gen.step()
        total_gen_loss += loss_c.item()
        total += 1
    timenow = timeSince(start)
    print ('Time %s, Epoch [%d/%d], D Loss: %.4f, G Loss: %.4f, Total Loss: %.4f' 
            %(timenow, epoch+1, NUM_EPOCHS, total_disc_loss/total, total_gen_loss/total, (total_disc_loss+total_gen_loss)/total))

    total_gen_loss = 0
    total_disc_loss = 0
    total = 0
    for img, label in val_loader:
        if img.size(0) < BATCH_SIZE: continue
        img = V(img).cuda()
        d = D(img)
        loss_a = 0.5 * -d.log().mean()
        seed = seed_distribution.sample()
        x_fake = G(seed)
        d = D(x_fake.detach())
        loss_b = 0.5 * -(1 - d + 1e-10).log().mean()
        total_disc_loss += loss_a.item() + loss_b.item()
        d = D(x_fake) # no detach here
        loss_c = (1 - d + 1e-10).log().mean()
        # loss_c = -(d + 1e-10).log().mean()
        total_gen_loss += loss_c.item()
        total += 1
    timenow = timeSince(start)
    print ('Val loop. Time %s, Epoch [%d/%d], D Loss: %.4f, G Loss: %.4f, Total Loss: %.4f' 
            %(timenow, epoch+1, NUM_EPOCHS, total_disc_loss/total, total_gen_loss/total, (total_disc_loss+total_gen_loss)/total))

    exit()
    assert(0==1) # sorry bro
    torch.save(G.state_dict(), args.gen_file)
    torch.save(D.state_dict(), args.disc_file)
    
    if args.track_space and (args.latent_dim == 2) and ((epoch%10 == 0) or (epoch == NUM_EPOCHS-1)):
        nx = ny = 20
        x_values = np.linspace(-2, 2, nx) # sasha suggests -2,2 and altosaar uses -3,3
        y_values = np.linspace(-2, 2, ny)
        canvas = np.empty((28 * ny, 28 * nx))
        for ii, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                np_z = np.array([[xi, yi]])
                x_mean = G(torch.cuda.FloatTensor(np_z))
                canvas[(nx - ii - 1) * 28:(nx - ii) * 28, j *
                       28:(j + 1) * 28] = x_mean[0].data.reshape(28, 28)
        # imsave('prior_predictive_map_frame_%d.png', canvas)
        plt.figure(figsize=(8, 10))
        Xi, Yi = np.meshgrid(x_values, y_values)
        plt.imshow(canvas, origin="upper")
        plt.tight_layout()
        plt.savefig('latent_space_viz_gan_epoch_{}_of_{}.png'.format(epoch,NUM_EPOCHS))
