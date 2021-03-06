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
parser.add_argument('--optimizer_type','-optim',type=int,default=0,help='SGD optimizer')
parser.add_argument('--batch_size','-bs',type=int,default=128,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=10,help='Number of epochs')
parser.add_argument('--learning_rate','-lr',type=float,default=0.0001,help='Learning rate')
parser.add_argument('--rho','-r',type=float,default=0.95,help='rho for Adadelta optimizer')
parser.add_argument('--alpha','-al',type=float,default=0.98,help='alpha value for RMSprop optimizer')
parser.add_argument('--weight_decay','-wd',type=float,default=0.0,help='Weight decay constant for optimizer')
parser.add_argument('--max_weight_norm','-wn',type=float,help='Max L2 norm for weight clippping')
parser.add_argument('--clip','-c',type=float,help='Max norm for weight clipping')
parser.add_argument('--model_file','-mf',type=str,default='/n/data_01/cs287-s18/projects/knn_00.pkl',help='Save model filename')
parser.add_argument('--stop_instance','-halt',action='store_true',help='Stop AWS instance after training run.')
parser.add_argument('--log_file','-l',type=str,default='stderr',help='training log file')
parser.add_argument('--cell_set','-cs',type=str,default='train')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
args = parser.parse_args()

print("Begin run")

if args.log_file == 'stderr':
    Logger = sys.stderr
else:
    log_file = open(args.log_file,'w')
    Logger = log_file
#Logger = sys.stderr

if args.model_type == 0:
    model = Basset(output_labels=56)
elif args.model_type == 1:
    model = DeepSEA(output_labels=56)
elif args.model_type == 2:
    model = Classic(output_labels=56)
elif args.model_type == 3:
    model = BassetNorm(output_labels=56)
elif args.model_type == 4:
    model = DanQ(output_labels=56)

model.cuda()
num_params = sum([p.numel() for p in model.parameters()])
print("Model successfully imported\nTotal number of parameters {}".format(num_params),file=Logger)

start = time.time()
print("Reading data from file {}".format(args.data),file=Logger)
data = h5py.File(args.data)

# Set celltype holdouts
# ['E004','E038','E082','E095','E098','E123','E127'] ['H1 Derived Mesendo','CD4 Naive Primary','Fetal Brain','Left Ventricle','Pancreas','K562','NHEK-Epidermal']
alltypes   = [ str(x, 'utf-8') for x in list(data['target_labels'][:]) ]
holdouts   = ['E004','E038','E082','E095','E098','E123','E127']

train_type = [ x for x in alltypes if x not in holdouts ]
valid_type = ['E004','E095','E098','E127']
test_type  = ['E038','E082','E123']

if args.cell_set == 'train':
    get_type = train_type
elif args.cell_set == 'valid':
    get_type = valid_type
elif args.cell_set == 'test':
    get_type = test_type
elif args.cell_set == 'all':
    get_type = alltypes
else:
    assert 0 == 1, "--cell_set flag must be 'train', 'valid', or 'test'"

c_idx = [ i for i,x in enumerate( list(data['target_labels'][:]) ) if str(x, 'utf-8') in get_type ]
if args.cell_set == 'all':
    v_idx = [ i for i,x in enumerate( list(data['target_labels'][:]) ) if str(x, 'utf-8') in valid_type ]
else:
    v_idx = c_idx


train = torch.utils.data.TensorDataset(torch.CharTensor(data['train_in'][:]), 
                                       torch.CharTensor(data['train_out'][:,c_idx]))
val = torch.utils.data.TensorDataset(torch.CharTensor(data['valid_in'][:]), 
                                     torch.CharTensor(data['valid_out'][:,c_idx]))
test = torch.utils.data.TensorDataset(torch.CharTensor(data['test_in'][:]), 
                                      torch.CharTensor(data['test_out'][:,c_idx]))
train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
# train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False)
# val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
print("Dataloaders generated {}".format( timeSince(start) ),file=Logger)

params = list(filter(lambda x: x.requires_grad, model.parameters()))
if args.optimizer_type == 0:
    optimizer = torch.optim.Adadelta(params, lr=args.learning_rate, rho=args.rho, weight_decay=args.weight_decay)
elif args.optimizer_type == 1:
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer_type == 2:
    optimizer = torch.optim.RMSprop(params, lr=args.learning_rate, alpha=args.alpha, weight_decay=args.weight_decay)

#criterion = torch.nn.MultiLabelSoftMarginLoss() # Loss function
criterion = torch.nn.BCEWithLogitsLoss(size_average=False)

start = time.time()
best_loss = np.inf
print("Begin training",file=Logger)
for epoch in range(args.num_epochs):
    model.train()
    #train_loader.init_epoch()
    ctr = 0
    tot_loss = 0
    for inputs, targets in train_loader:
        inputs = to_one_hot(inputs, n_dims=4).permute(0,3,1,2).squeeze().float()
        targets = targets.float()
        inp_batch = Variable(inputs).cuda()
        trg_batch = Variable(targets).cuda()        
        optimizer.zero_grad()
        outputs = model(inp_batch)
        loss = criterion(outputs.view(-1), trg_batch.view(-1))
        loss.backward()
        tot_loss += loss.item()
        if args.clip:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        if args.max_weight_norm:
            model.clip_norms( args.max_weight_norm )
        ctr += 1
        if ctr % 100 == 0:
            timenow = timeSince(start)
            print('Epoch [{}/{}], Iter [{}/{}], Time: {}, Loss: {}'.format(epoch+1, args.num_epochs, ctr,
                                                                           len(train)//args.batch_size, 
                                                                           timenow, tot_loss/(100*args.batch_size)),
                  file=Logger)
            tot_loss = 0
    #
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
    avg_ROC_auc = calc_auc(model, np.row_stack(y_test)[:,v_idx], np.row_stack(y_score)[:,v_idx],"ROC")
    avg_PR_auc = calc_auc(model, np.row_stack(y_test)[:,v_idx], np.row_stack(y_score)[:,v_idx],"PR")
    timenow = timeSince(start)
    print( "Epoch [{}/{}], Time: {}, Validation loss: {}, Mean ROC AUC: {}, Mean PRAUC: {}".format( epoch+1, args.num_epochs, 
                                                                                timenow, epoch_loss, avg_ROC_auc, avg_PR_auc),
           file=Logger)
    if epoch_loss <= best_loss:
        torch.save(model.state_dict(), args.model_file)
        print( "Delta loss: {}, Model saved at {}".format((epoch_loss-best_loss),args.model_file) , file=Logger)
        best_loss = epoch_loss

if args.stop_instance:
    Logger.close()
    subprocess.call(['sudo','halt'])
