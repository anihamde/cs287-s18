import time
import numpy as np
from collections import Counter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alphatri','-atr',type=float, default=0.33)
parser.add_argument('--alphannlm','-an',type=float, default=0.33)
parser.add_argument('--nnlmpath','-np',type=str,default='../../models/HW2/nnlm.pkl',help='Path to nnlm model.')
parser.add_argument('--lstmpath','-lp',type=str,default='../../models/HW2/lstm.pkl',help='Path to lstm model.')

parser.add_argument('--alphab','-ab',type=float, default=0.4)
parser.add_argument('--alphat','-at',type=float, default=0.25)

parser.add_argument('--batch_size','-bs',type=int,default=10,help='set training batch size. default = 10.')
parser.add_argument('--receptive_field','-rf',type=int,default=5,help='set receptive field of nnlm.')
parser.add_argument('--nnlm_hidden_size','-nhs',type=int,default=100,help='set size of hidden layer.')
parser.add_argument('--dropout_rate','-dr',type=float,default=0.5,help='set dropout rate for deep layers.')
parser.add_argument('--embedding_max_norm','-emn',type=float,default=15,help='set max L2 norm of word embedding vector.')

parser.add_argument('--num_layers','-nl',type=int,default=2,help='set number of lstm layers.')
parser.add_argument('--lstm_hidden_size','-lhs',type=int,default=500,help='set size of hidden layer.')
parser.add_argument('--clip_constraint','-c',type=float,default=5,help='set constraint for gradient clipping.')
args = parser.parse_args()

alphatri = args.alphatri
alphannlm = args.alphannlm

# Hyperparameters
alpha_b = args.alphab # confusing, i know
alpha_t = args.alphat

bs = args.batch_size # batch size
n = args.receptive_field # receptive field
n_hidden_size = args.nnlm_hidden_size
dropout_rate = args.dropout_rate
emb_mn = args.embedding_max_norm

n_layers = args.num_layers
l_hidden_size = args.lstm_hidden_size
constraint = args.clip_constraint

# Text processing library
import torchtext
from torchtext.vocab import Vectors
# Our input $x$
TEXT = torchtext.data.Field()
# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="./", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

TEXT.build_vocab(train)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=bs, device=-1, bptt_len=32, repeat=False)

url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) # feel free to alter path
word2vec = TEXT.vocab.vectors
print("REMINDER!!! Did you create ../../models/HW2?????")






uni = Counter()
bi = Counter()
tri = Counter()
biprev = [1] * bs # "1" is <pad>
triprev = [1] * bs * 2

for batch in iter(train_iter):
    txt = batch.text.data
    uni.update(txt.view(-1).tolist()) # throw all words into bag
    bi0 = biprev + txt[:-1,:].view(-1).tolist()
    bi1 = txt.view(-1).tolist()
    biprev = txt[-1,:].view(-1).tolist()
    bi.update(zip(bi0,bi1))
    tri0 = triprev + txt[:-2,:].view(-1).tolist()
    tri1 = triprev[bs:] + txt[:-1,:].view(-1).tolist()
    tri2 = txt.view(-1).tolist()
    triprev = txt[-2:,:].view(-1).tolist()
    tri.update(zip(tri0,tri1,tri2))

print("Done training trigram!")
# TODO: experiment with ignoring EOS for unigrams, like this
# uni[TEXT.vocab.stoi["<eos>"]] = 0

unisum = sum(uni.values()) # just normalize once for unigrams
for k in uni:
    uni[k] *= (1 - alpha_b - alpha_t) / unisum
print("Done normalizing unigram counter!")

def predict(l):
    # filter
    bifilt = Counter({k:bi[k] for k in bi if k[0]==l[-1]})
    trifilt = Counter({k:tri[k] for k in tri if k[0]==l[-2] and k[1]==l[-1]})
    # normalize
    bisum = sum(bifilt.values())
    trisum = sum(trifilt.values())
    # combine
    total = copy.copy(uni) # shallow copy
    for k in bifilt:
        total[k[-1]] += bifilt[k] * alpha_b / bisum
    for k in trifilt:
        total[k[-1]] += trifilt[k] * alpha_t / trisum
    return total

# here's batch size is wrong
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        # Test the max_norm. Is it norm per row, or total norm of the whole matrix?
        self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1),max_norm=emb_mn)
        self.embeddings.weight.data = TEXT.vocab.vectors
        self.h = nn.Linear(n*300,n_hidden_size)
        self.u = nn.Linear(n_hidden_size,len(TEXT.vocab))
        self.w = nn.Linear(n*300,len(TEXT.vocab))
    def forward(self, inputs): # inputs (batch size, "sentence" length) bs,n
        embeds = self.embeddings(inputs) # bs,n,300
        embeds = embeds.view(-1,n*300) # bs,n*300
        out = F.tanh(self.h(embeds)) # bs,hidden_size
        out = self.u(F.dropout(out,p=dropout_rate)) # bs,|V|
        embeds = F.dropout(embeds,p=dropout_rate)
        out += self.w(embeds) # bs,|V|
        #out = F.softmax(out,dim=1)
        return out

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class dLSTM(nn.Module):
    def __init__(self):
        super(dLSTM, self).__init__()
        self.embedding = nn.Embedding(word2vec.size(0),word2vec.size(1),max_norm=emb_mn)
        self.embedding.weight.data.copy_(word2vec)
        self.lstm = nn.LSTM(word2vec.size(1), l_hidden_size, n_layers, dropout=dropout_rate)
        self.linear = nn.Linear(l_hidden_size, len(TEXT.vocab))
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, input, hidden): 
        # input is (sentence length, batch size) n,bs
        # hidden is ((n_layers,bs,hidden_size),(n_layers,bs,hidden_size))
        embeds = self.embedding(input) # n,bs,300
        # batch goes along the second dimension
        out = F.dropout(embeds,p=dropout_rate)
        out, hidden = self.lstm(out, hidden)
        out = F.dropout(out,p=dropout_rate)
        # apply the linear and the softmax
        out = self.linear(out) # n,bs,|V|
        #out = self.softmax(out)    # This was originally the output. (SG: I see this is LogSoftmax)
        return out, hidden
    
    def initHidden(self):
        h0 = torch.zeros(n_layers, bs, hidden_size).type(torch.FloatTensor)
        c0 = torch.zeros(n_layers, bs, hidden_size).type(torch.FloatTensor)
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (Variable(h0), Variable(c0))




nnlm = NNLM()
lstm = dLSTM()
if torch.cuda.is_available():
    nnlm.cuda()
    lstm.cuda()

nnlm.load_state_dict(torch.load(args.nnlmpath))
lstm.load_state_dict(torch.load(args.lstmpath))

# Evaluator
correct = total = 0
precisionmat = (1/np.arange(1,21))[::-1].cumsum()[::-1]
precision = 0
crossentropy = 0

hidden = model.initHidden()
for batch in iter(val_iter):
    sentences = batch.text.data # n,bs
    if sentences.size(0) < 3: # make sure sentence length is long enough
        next # don't worry about it

    lsent = sentences.cuda()
    labels = lsent[-1] # bs
    lout, hidden = model(sentences, hidden)
    lout = lout[-2] # bs,|V|
    
    nsent = lsent.transpose(1,0) # bs,n
    nout = model(nsent[:,-1-n:-1]) # bs,|V|
    labels = nsent[:,-1] # bs
    # apply softmaxes
    out = alphannlm * nout + (1-alphatri-alphannlm) * lout
    for s in range(bs):
        tout = predict(sentences[-3:-1,s]) # counter
        # add to combined matrix
        # tradeoff accuracy for speed?
        for word in tout:
            out[s,word] += alphatri * tout[word]
    crossentropy += F.nll_loss(out,labels)
    out, labels = out.data, labels.data
    _, outsort = torch.sort(out,dim=1,descending=True)
    outsort = outsort[:,:20]
    inds = (outsort-labels.unsqueeze(1)==0)
    inds = inds.sum(dim=0).type(torch.cuda.FloatTensor)
    precision += inds.dot(precisionmat)
    # plain ol accuracy
    _, predicted = torch.max(out, 1)
    total += labels.ne(padidx).int().sum()
    correct += (predicted==labels).sum()
    hidden = repackage_hidden(hidden)

    print('Test Accuracy', correct/total)
    print('Precision',precision/total)
    print('Perplexity',torch.exp(bs*crossentropy/total).data[0])


