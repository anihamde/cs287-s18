import numpy as np
from collections import Counter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
bs = 10 # batch size
n = 3 # number of words

# Text processing library
import torchtext
from torchtext.vocab import Vectors
# Our input $x$
TEXT = torchtext.data.Field()
# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="./", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

print('len(train)', len(train))

TEXT.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))

# TODO: use argparse
if False:
    TEXT.build_vocab(train, max_size=1000)
    len(TEXT.vocab)

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=bs, device=-1, bptt_len=32, repeat=False)
# TODO: can we try bptt_len as 10? (it would give us more data)
# or make it n, actually?

it = iter(train_iter)
batch = next(it) 
print("Size of text batch [max bptt length, batch size]", batch.text.size())
print("Second in batch", batch.text[:, 2])
print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 2].data]))
batch = next(it)
print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 2].data]))

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('../HW1/wiki.simple.vec', url=url)) # feel free to alter path
print("Word embeddings size ", TEXT.vocab.vectors.size())

# TODO: mixture of models with interpolated trigram (fixed or learned weights)
# TODO: weight decay penalty (excluding bias terms)
# TODO: bengio extensions
# TODO: parallel computation (make everything cuda)
# TODO: our own extensions (multichannel with glove, static/dynamic, etc?) (conv layers) (dropout) (recurrence)

class NNLM(nn.Module):
    def __init__(self, input_size):
        super(NNLM, self).__init__()
        self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
        self.embeddings.weight.data = TEXT.vocab.vectors
        self.h = nn.Linear(n*300,100) # just made up a hidden dimension
        self.u = nn.Linear(100,len(TEXT.vocab))
        self.w = nn.Linear(n*300,len(TEXT.vocab))

    def forward(self, inputs): # inputs (bs,words/sentence) 10,32
        bsz = inputs.size(0) # batch size might change
        if inputs.size(1) < n: # padding issues on really short sentences
            pads = Variable(torch.zeros(bsz,n-inputs.size(1))).type(torch.LongTensor)
            inputs = torch.cat([pads,inputs],dim=1)
        embeds = self.embeddings(inputs[:,-n:]) # bsz,n,300
        embeds = embeds.view(bsz,-1) # bsz, n*300
        out = F.tanh(self.h(embeds)) # bsz, 100
        out = self.u(out) # bsz, |V|
        out += self.w(embeds) # bsz, |V|
        out = F.softmax(out,dim=1)
        return out