import numpy as np
from collections import Counter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
bs = 10 # batch size
learning_rate = .001
weight_decay = 0
num_epochs = 20 # idk!

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
word2vec = TEXT.vocab.vectors()


# TODO: dropout, and model.train()/eval()
# TODO: learning rate decay? (zaremba has specific instructions for this)


hidden_size = 3
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(word2vec.size(0), word2vec.size(1))
        self.i2h = nn.Linear(word2vec.size(1)+hidden_size, hidden_size)
        self.i2o = nn.Linear(word2vec.size(1)+hidden_size, len(TEXT.vocab))
        self.softmax = nn.LogSoftmax(dim=1) # TODO: does this line still throw an error?

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        combined = torch.cat((embedded, hidden), 1)
        hidden = self.i2h(combined)
        out = self.i2o(combined)
        out = self.softmax(out)
        return out, hidden

model = RNN()
criterion = nn.NLLLoss()
params = filter(lambda x: x.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

losses = []
model.train()
for i in range(num_epochs):
    train_iter.init_epoch()
    ctr = 0
    # initialize hidden vector
    hidden = Variable(torch.zeros(bs, hidden_size))
    for batch in iter(train_iter):
        model.zero_grad() # TODO: will this kill my saved hidden from prev batch?
        # calculate forward pass, except for very last word
        for i in range(batch.size(1)-1):
            out, hidden = model(batch.text[:, i], hidden)
        loss = criterion(out, batch.text[:,-1])
        loss.backward()
        optimizer.step()
        # TODO: weight clippings nn.utils.clip_grad_norm(params, constraint) is this right?
        # update hidden vector
        _, hidden = model(batch.text[:, -1], hidden)
        ctr += 1
        if ctr % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                %(epoch+1, num_epochs, ctr, len(train)//bs, loss.data[0]))
        losses.append(loss.data[0])

# TODO: code to save numpy losses, model pkl, or load model

model.eval()
correct = total = 0
hidden = Variable(torch.zeros(bs, hidden_size))
for batch in iter(val_iter):
    # calculate forward pass, except for very last word
    for i in range(batch.size(1)-1):
        out, hidden = model(batch.text[:,i], hidden)
    _, predicted = torch.max(out.data, 1)
    labels = batch.text[:,-1]
    total += labels.size(0)
    correct += (predicted == labels).sum()
    _, hidden = model(batch.text[:,-1], hidden)

print('Test Accuracy', correct/total)

# TODO: better loss measurements (top 20 precision, perplexity)
# TODO: write results to csv