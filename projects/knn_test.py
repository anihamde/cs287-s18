import torch
import h5py
import sys
import subprocess
import glob
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
parser.add_argument('--model_type','-mt',type=int,default=3,help='Model type')
parser.add_argument('--batch_size','-bs',type=int,default=128,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=10,help='Number of epochs')
parser.add_argument('--model_file','-mf',type=str,default='../../model_from_lua.pkl',help='Load model filename')
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
    model = Basset(output_labels=49)
elif args.model_type == 1:
    model = DeepSEA(output_labels=49)
elif args.model_type == 2:
    model = Classic(output_labels=49)
elif args.model_type == 3:
    model = BassetNorm(output_labels=49)

model.load_state_dict(torch.load(args.model_file))
model.cuda()
num_params = sum([p.numel() for p in model.parameters()])
print("Model {} successfully imported\nTotal number of parameters {}".format(mf, num_params),file=Logger)

expn_pth = '/n/data_02/Basset/data/expn/roadmap/57epigenomes.RPKM.pc'
print("Reading gene expression data from:\n{}".format(expn_pth))
# Gene expression dataset
expn = pd.read_table(expn_pth,header=0)    
col_names = expn.columns.values[1:]
expn = expn.drop(col_names[-1],axis=1)
expn.columns = col_names
pinned_lookup = torch.nn.Embedding.from_pretrained(torch.FloatTensor(expn.as_matrix().T),freeze=True)
pinned_lookup.cuda()
print("Done")
# nicer euclidean similarity matrix at https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/7
def similarity_matrix(mat):
    simmat = torch.zeros(mat.size(0),mat.size(0))
    for i in range(mat.size(0)):
        for j in range(i,mat.size(0)):
            simmat[i,j] = simmat[j,i] = F.cosine_similarity(mat[i],mat[j],dim=0)[0]
    return simmat

simmat = similarity_matrix(pinned_lookup.weight.data)
k_weights, k_nearest = simmat.sort(descending=True)
k = 2
k_weights, k_nearest = k_weights[:,1:k+1], k_nearest[:,1:k+1]
k_weights = F.normalize(k_weights, dim=1)
mult_tensor = torch.zeros(57,57)
mult_tensor.scatter(dim=1, k_nearest, k_weights)

print("Reading data from file {}".format(args.data),file=Logger)
data = h5py.File(args.data)

# Set celltype holdouts
# ['E004','E038','E082','E095','E098','E123','E127'] ['H1 Derived Mesendo','CD4 Naive Primary','Fetal Brain','Left Ventricle','Pancreas','K562','NHEK-Epidermal']
alltypes   = [ str(x, 'utf-8') for x in list(data['target_labels'][:]) ]
holdouts   = ['E004','E038','E082','E095','E098','E123','E127']

train_type = [ x for x in alltypes if x not in holdouts ]
valid_type = ['E004','E095','E098','E127']
test_type  = ['E038','E082','E123']

c_idx = [ i for i,x in enumerate( list(data['target_labels'][:]) ) if str(x, 'utf-8') in train_type ]

train = torch.utils.data.TensorDataset(torch.CharTensor(data['train_in'][:,c_idx]), 
                                       torch.CharTensor(data['train_out'][:,c_idx]))
val = torch.utils.data.TensorDataset(torch.CharTensor(data['valid_in'][:,c_idx]), 
                                     torch.CharTensor(data['valid_out'][:,c_idx]))
test = torch.utils.data.TensorDataset(torch.CharTensor(data['test_in'][:,c_idx]), 
                                      torch.CharTensor(data['test_out'][:,c_idx]))
train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
# train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False)
# val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
print("Dataloaders generated {}".format( timeSince(start) ),file=Logger)

#criterion = torch.nn.MultiLabelSoftMarginLoss() # Loss function
criterion = torch.nn.BCEWithLogitsLoss(size_average=False)

start = time.time()
print("Dataloaders generated {}".format( timeSince(start) ),file=Logger)

# model_files = sorted(glob.glob('bassetnorm_*.pkl'))
# for mf in model_files:

model.eval()
losses  = []
y_score = []
y_test  = []
#val_loader.init_epoch()
for inputs, geneexpr, targets in test_loader:
    # geneexpr_batch = pinned_lookup(geneexpr.long().cuda()).squeeze()
    inputs = to_one_hot(inputs, n_dims=4).permute(0,3,1,2).squeeze().float()
    targets = targets.float()
    inp_batch = Variable( inputs ).cuda()
    trg_batch = Variable( targets ).cuda()        
    moutputs = model(inp_batch) # ?, 49
    # can i get some things?
    outputs = moutputs * mult_tensor[geneexpr.long().cuda().squeeze()]
    loss = criterion(outputs.view(-1), trg_batch.view(-1))
    losses.append(loss.item())
    y_score.append( outputs.cpu().data.numpy() )
    y_test.append(  targets.cpu().data.numpy() )

epoch_loss = sum(losses)/len(val)
avg_auc = calc_auc(model, np.row_stack(y_test), np.row_stack(y_score))
timenow = timeSince(start)
print( "Time: {}, Validation loss: {}, Mean AUC: {}".format( timenow, epoch_loss, avg_auc),
       file=Logger)

if args.stop_instance:
    Logger.close()
    subprocess.call(['sudo','halt'])

