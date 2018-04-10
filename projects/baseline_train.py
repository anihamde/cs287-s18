import torch
import h5py
import sys
import subprocess
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

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--data','-d',type=str,default='/n/data_02/Basset/data/mini_roadmap.h5',help='path to training data')
parser.add_argument('--model_type','-mt',type=int,default=0,help='Model type')
parser.add_argument('--batch_size','-bs',type=int,default=100,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=10,help='Number of epochs')
parser.add_argument('--learning_rate','-lr',type=float,default=0.0001,help='Learning rate')
parser.add_argument('--rho','-r',type=float,default=0.95,help='rho for Adadelta optimizer')
parser.add_argument('--weight_decay','-wd',type=float,default=0.0,help='Weight decay constant for optimizer')
parser.add_argument('--model_file','-mf',type=str,default='stupid.pkl',help='Save model filename')
parser.add_argument('--stop_instance','-halt',action='store_true',help='Stop AWS instance after training run.')
args = parser.parse_args()

print("Begin run")

if args.stop_instance:
    log_file = open('.log_file.txt','w')
    Logger = log_file
else:
    Logger = sys.stderr

start = time.time()
print("Reading data from file {}".format(args.data),file=Logger)
data = h5py.File(args.data)

train = torch.utils.data.TensorDataset(torch.CharTensor(data['train_in']), torch.CharTensor(data['train_out']))
val = torch.utils.data.TensorDataset(torch.CharTensor(data['valid_in']), torch.CharTensor(data['valid_out']))
test = torch.utils.data.TensorDataset(torch.CharTensor(data['test_in']), torch.CharTensor(data['test_out']))
train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True)
print("Dataloaders generated {}".format( timeSince(start) ),file=Logger)

if model_type == 0:
    model = Basset()
elif model_type == 1:
    model = DeepSEA()

model.cuda()
print("Model successfully imported",file=Logger)


params = list(filter(lambda x: x.requires_grad, model.parameters()))
optimizer = torch.optim.Adadelta(params, lr=args.learning_rate, rho=args.rho, weight_decay=args.weight_decay)
#criterion = torch.nn.MultiLabelSoftMarginLoss() # Loss function
criterion = torch.nn.BCEWithLogitsLoss()

start = time.time()
last_loss = 100000
print("Begin training",file=Logger)
for epoch in range(args.num_epochs):
    model.train()
    #train_loader.init_epoch()
    ctr = 0
    for inputs, targets in train_loader:
        inputs = to_one_hot(inputs, n_dims=4).permute(0,3,1,2).squeeze().float()
        targets = targets.float()
        inp_batch = Variable(inputs).cuda()
        trg_batch = Variable(targets).cuda()        
        optimizer.zero_grad()
        outputs = model(inp_batch)
        loss = criterion(outputs.view(-1), trg_batch.view(-1))
        loss.backward()
        optimizer.step()
        ctr += 1
        if ctr % 100 == 0:
            timenow = timeSince(start)
            print('Epoch [{}/{}], Iter [{}/{}], Time: {}, Loss: {}'.format(epoch+1, args.num_epochs, ctr,
                                                                           len(train)//args.batch_size, 
                                                                           timenow, loss.item()),
                  file=Logger)
    #
    model.eval()
    losses = []
    #val_loader.init_epoch()
    for inputs, targets in val_loader:
        inputs = to_one_hot(inputs, n_dims=4).permute(0,3,1,2).squeeze().float()
        targets = targets.float()
        inp_batch = Variable( inputs ).cuda()
        trg_batch = Variable(targets).cuda()        
        outputs = model(inp_batch)
        loss = criterion(outputs.view(-1), trg_batch.view(-1))
        losses.append(loss.item())
    epoch_loss = sum(losses)/float(len(losses))
    timenow = timeSince(start)
    print( "Epoch [{}/{}], Time: {}, Validation loss: {}".format( epoch+1, args.num_epochs, timenow, epoch_loss),
           file=Logger)
    if epoch_loss <= last_loss:
        torch.save(model.state_dict(), args.model_file)
        print( "Delta loss: {}, Model saved at {}".format((epoch_loss-last_loss),args.model_file) , file=Logger)
    last_loss = epoch_loss

if args.stop_instance:
    Logger.close()
    subprocess.call(['sudo','halt'])