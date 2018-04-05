import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.distributions import Normal
import numpy as np
import argparse
import time
from helpers import timeSince, asMinutes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from baseline_model import *
import h5py

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--data','-d',type=str,default='/n/data_02/Basset/data/encode_roadmap.h5',help='path to training data')
parser.add_argument('--model_type','-mt',type=int,default=0,help='Model type')
parser.add_argument('--batch_size','-bs',type=int,default=100,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=50,help='Number of epochs')
parser.add_argument('--learning_rate','-lr',type=float,default=0.0001,help='Learning rate')
parser.add_argument('--rho','-r',type=float,default=0.95,help='rho for Adadelta optimizer')
parser.add_argument('--weight_decay','-wd',type=float,default=0.0,help='Weight decay constant for optimizer')
parser.add_argument('--model_file','-mf',type=str,default='stupid.pkl',help='Save model filename')
args = parser.parse_args()

data = h5py.File(args.data)
print("Reading data from file {}".format(args.data))

train = torch.utils.data.TensorDataset(torch.LongTensor(data['train_in']), torch.LongTensor(data['train_out']))
val = torch.utils.data.TensorDataset(torch.LongTensor(data['valid_in']), torch.LongTensor(data['valid_out']))
test = torch.utils.data.TensorDataset(torch.LongTensor(data['test_in']), torch.LongTensor(data['test_out']))
train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True)
print("Dataloaders generated")
      
model = Basset()
print("Model successfully imported")


params = list(filter(lambda x: x.requires_grad, model.parameters()))
optimizer = optim.Adadelta(params, lr=learning_rate, rho=args.rho, weight_decay=args.weight_decay)

start = time.time()
last_loss = 100000
print("Go go go: {}".format(asMinutes(start)))
for epoch in range(num_epochs):
    model.train()
    train_iter.init_epoch()
    ctr = 0
    for inputs, targets in train_loader:
        inp_batch = Variable(inputs).cuda()
        trg_batch = Variable(targets).cuda()        
        optimizer.zero_grad()
        outputs = model(inp_batch)
        loss = criterion(outputs, trg_batch)
        loss.backward()
        optimizer.step()
        ctr += 1
        if ctr % 100 == 0:
            timenow = timeSince(start)
            print('Epoch [{}/{}], Iter [{}/{}], Time: {}, Loss: {}'.format(epoch+1, num_epochs, ctr,
                                                                           len(train)//args.batch_size, 
                                                                           timenow, loss.item()))
        losses.append(loss.item())
    #
    model.eval()
    losses = []
    for inputs, targets in val_loader:
        inp_batch = Variable(inputs).cuda()
        trg_batch = Variable(targets).cuda()        
        outputs = model(inp_batch)
        loss = criterion(outputs, trg_batch)
        losses.append(loss.item())
    epoch_loss = sum(losses)
    timenow = timeSince(start)
    print( "Epoch [{}/{}], Time: {}, Validation loss: {}".format( epoch, num_epochs, timenow, epoch_loss) )
    if epoch_loss <= last_loss:
        torch.save(model.state_dict(), args.model_file)
        print( "Model saved at {}".format(args.model_file) )
