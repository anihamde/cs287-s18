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
from helpers import timeSince, asMinutes, calc_auc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from baseline_model import *

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--data','-d',type=str,default='/n/data_02/Basset/data/mini_roadmap.h5',help='path to training data')
parser.add_argument('--model_type','-mt',type=int,default=0,help='Model type')
parser.add_argument('--batch_size','-bs',type=int,default=128,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=10,help='Number of epochs')
parser.add_argument('--max_weight_norm','-wn',type=float,help='Max L2 norm for weight clippping')
parser.add_argument('--clip','-c',type=float,help='Max norm for weight clipping')
parser.add_argument('--model_file','-mf',type=str,default='stupid.pkl',help='Save model filename')
parser.add_argument('--stop_instance','-halt',action='store_true',help='Stop AWS instance after training run.')
parser.add_argument('--log_file','-l',type=str,default='stderr',help='training log file')
args = parser.parse_args()


print("Begin run")

if args.log_file == 'stderr':
    Logger = sys.stderr
else:
    log_file = open(args.log_file,'w')
    Logger = log_file
#Logger = sys.stderr

if args.model_type == 0:
    model = Basset()
elif args.model_type == 1:
    model = DeepSEA()
elif args.model_type == 2:
    model = Classic()
elif args.model_type == 3:
    model = BassetNorm()

num_params = sum([p.numel() for p in model.parameters()])
    
model.cuda()
print("Model successfully imported\nTotal number of parameters {}".format(num_params),file=Logger)

start = time.time()
print("Reading data from file {}".format(args.data),file=Logger)
data = h5py.File(args.data)

# train = torch.utils.data.TensorDataset(torch.CharTensor(data['train_in']), torch.CharTensor(data['train_out']))
val = torch.utils.data.TensorDataset(torch.CharTensor(data['valid_in']), torch.CharTensor(data['valid_out']))
test = torch.utils.data.TensorDataset(torch.CharTensor(data['test_in']), torch.CharTensor(data['test_out']))
# train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False)
print("Dataloaders generated {}".format( timeSince(start) ),file=Logger)


#criterion = torch.nn.MultiLabelSoftMarginLoss() # Loss function
criterion = torch.nn.BCEWithLogitsLoss(size_average=False)

start = time.time()
best_loss = np.inf
print("Begin training",file=Logger)

model.eval()
losses  = []
y_score = []
y_test  = []
#val_loader.init_epoch()
for inputs, targets in val_loader:
    inputs = to_one_hot(inputs, n_dims=4).permute(0,3,1,2).squeeze().float()
    targets = targets.float()
    inp_batch = Variable( inputs ).cuda()
    trg_batch = Variable(targets).cuda()        
    outputs = model(inp_batch)
    loss = criterion(outputs.view(-1), trg_batch.view(-1))
    losses.append(loss.item())
    y_score.append( outputs.cpu().data.numpy() )
    y_test.append(  targets.cpu().data.numpy() )
epoch_loss = sum(losses)/len(val)
avg_auc = calc_auc(model, np.row_stack(y_test), np.row_stack(y_score))
timenow = timeSince(start)
print( "Epoch [{}/{}], Time: {}, Validation loss: {}, Mean AUC: {}".format( epoch+1, args.num_epochs, 
                                                                            timenow, epoch_loss, avg_auc),
       file=Logger)

if args.stop_instance:
    Logger.close()
    subprocess.call(['sudo','halt'])

