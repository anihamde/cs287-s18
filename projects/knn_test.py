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
import pandas as pd
import argparse
import time
from helpers import timeSince, asMinutes, calc_auc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from baseline_model import *

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--data','-d',type=str,default='/n/data_02/Basset/data/mini_roadmap.h5',help='path to training data')
parser.add_argument('--model_type','-mt',type=int,default=3,help='Model type')
# parser.add_argument('--batch_size','-bs',type=int,default=128,help='Batch size')
parser.add_argument('--num_epochs','-ne',type=int,default=10,help='Number of epochs')
parser.add_argument('--model_file','-mf',type=str,default='/n/data_01/cs287-s18/projects/knn_00.pkl',help='Load model filename')
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

print("Reading data from file {}".format(args.data),file=Logger)
data = h5py.File(args.data)

# Set celltype holdouts
# ['E004','E038','E082','E095','E098','E123','E127'] ['H1 Derived Mesendo','CD4 Naive Primary','Fetal Brain','Left Ventricle','Pancreas','K562','NHEK-Epidermal']
alltypes   = [ str(x, 'utf-8') for x in list(data['target_labels'][:]) ]
holdouts   = ['E004','E038','E082','E095','E098','E123','E127']

train_type = [ x for x in alltypes if x not in holdouts ]
valid_type = ['E004','E095','E098','E127']
test_type  = ['E038','E082','E123']

# train = RoadmapDataset(data,expn,train_type,segment='train')
val   = RoadmapDataset(data,expn,valid_type,segment='valid')
test  = RoadmapDataset(data,expn,test_type,segment='test')
# Set Loader
# train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=500, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=500, shuffle=False)
print("Dataloaders generated {}".format( timeSince(start) ),file=Logger)

# nicer euclidean similarity matrix at https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/7
def similarity_matrix(mat):
    simmat = torch.zeros(mat.size(0),mat.size(0))
    for i in range(mat.size(0)):
        for j in range(i,mat.size(0)):
            simmat[i,j] = simmat[j,i] = F.cosine_similarity(mat[i],mat[j],dim=0).item()
    return simmat

valtestdex = np.concatenate([val.expn_dex,test.expn_dex])
traindex = np.setdiff1d(np.arange(57),valtestdex)
simmat = torch.zeros(7,49)
mat = pinned_lookup.weight
for i in range(7):
    for j in range(49):
        simmat[i,j] = F.cosine_similarity(mat[valtestdex[i]],mat[traindex[j]],dim=0).item()

k_weights, k_nearest = simmat.sort(descending=True)
k = 2
k_weights, k_nearest = k_weights[:,:k], k_nearest[:,:k]
k_weights = F.normalize(k_weights, dim=1)
tensor1 = torch.zeros(7,49)
tensor1.scatter_(1, k_nearest, k_weights)
tensor2 = torch.zeros(57,49)
tensor2[valtestdex,:] = tensor1
tensor2 = tensor2.cuda()
# take 7,49 thing and make it bigger so easy to index into with geneexpr

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
    outputs = moutputs * tensor2[geneexpr.long().cuda().squeeze()]
    outputs = outputs.sum(dim=1)
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

