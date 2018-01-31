# Text processing library and methods for pretrained word embeddings
import torchtext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext.vocab import Vectors, GloVe
import numpy as np
import matplotlib.pyplot as plt
import csv

# Our input $x$
TEXT = torchtext.data.Field()
# Our labels $y$
LABEL = torchtext.data.Field(sequential=False)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

TEXT.build_vocab(train)
LABEL.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))
print('len(LABEL.vocab)', len(LABEL.vocab))

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=-1, repeat=False)

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

print("Word embeddings size ", TEXT.vocab.vectors.size())

#########################################

orig = TEXT.vocab.vectors
sd = torch.load('../../models/cnn_standard.pkl')
new = sd['embeddings.weight']

# sim = F.cosine_similarity(orig,new,dim=1)
sim = torch.abs(orig-new).sum(1) # l1 distance
# sim = ((orig-new)**2).sum(1).sqrt() # l2 distance
sim = sim.numpy() # vector of 16286 similarity scores

keywords = ['direct','directs','produce','produces','vision','spirit','capture','captures',
    'character','actor','role','plot','style']
keysim = []
for i in range(len(keywords)):
    key = TEXT.vocab.stoi[keywords[i]]
    keysim.append(sim[key])
    print(keywords[i],sim[key])

plt.hist(sim)
plt.hist(keysim*1000) # for scale
plt.title('Histogram of distances between embedding vectors pre/post-training')
plt.legend(['true vocab','keywords (not2scale)'])
plt.show()

# If we have time, we could add a couple more keywords and do a hypothesis test