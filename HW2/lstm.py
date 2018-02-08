import numpy as np
from collections import Counter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv

# Hyperparameters
bs = 10 # batch size
hidden_size = 20
n_layers = 2
learning_rate = .001
weight_decay = 0
num_epochs = 20 # idk!
dropout_rate = 0.5

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

# TODO: learning rate decay? (zaremba has specific instructions for this)
# TODO: multichannel tests (with glove and stuff)
# TODO: bidirectional
# TODO: parallelize
# TODO: replace dropouts with functional dropouts

class dLSTM(nn.Module):
    def __init__(self):
        super(dLSTM, self).__init__()
        self.embedding = nn.Embedding(word2vec.size(0),word2vec.size(1))
        self.embeddings.weight.data.copy_(word2vec)
        self.dropbottom = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(word2vec.size(1), hidden_size, n_layers, dropout=dropout_rate)
        self.droptop = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, len(TEXT.vocab))
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden): 
        # input is (sentence length, batch size) n,bs
        # hidden is ((n_layers,bs,hidden_size),(n_layers,bs,hidden_size))
        embeds = self.embedding(input) # n,bs,300
        # batch goes along the second dimension
        out = self.dropbottom(embeds)
        out, hidden = self.lstm(out, hidden)
        out = self.droptop(out)
        # apply the linear and the softmax
        out = self.softmax(self.linear(out)) # n,bs,|V|
        return out, hidden

model = dLSTM()
criterion = nn.NLLLoss()
params = filter(lambda x: x.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

losses = []
model.train()
for i in range(num_epochs):
    train_iter.init_epoch()
    ctr = 0
    # initialize hidden vector
    hidden = (Variable(torch.zeros(n_layers, bs, hidden_size)), 
        Variable(torch.zeros(n_layers, bs, hidden_size)))
    for batch in iter(train_iter):
        sentences = batch.text # Variable of LongTensor of size (n,bs)
        out, hidden = model(sentences, hidden)
        loss = 0
        for i in range(sentences.size(1)-1):
            loss += criterion(out[i], sentences[i+1])
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # TODO: weight clippings nn.utils.clip_grad_norm(params, constraint) is this right?
        # hidden vector is automatically saved for next batch
        ctr += 1
        if ctr % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                %(epoch+1, num_epochs, ctr, len(train_iter), loss.data[0]))
        losses.append(loss.data[0])

    # can add a net_flag to these file names. and feel free to change the paths
    np.save("../../models/HW2/lstm_losses",np.array(losses))
    torch.save(model.state_dict(), '../../models/HW2/lstm.pkl')

# model.load_state_dict(torch.load('../../models/lstm.pkl'))

model.eval()
correct = total = 0
hidden = (Variable(torch.zeros(n_layers, bs, hidden_size)), 
        Variable(torch.zeros(n_layers, bs, hidden_size)))
for batch in iter(val_iter):
    sentences = batch.text
    out, hidden = model(sentences, hidden)
    _, predicted = torch.max(out.data, 2)
    labels = sentences[1:]
    total += labels.size(0) * labels.size(1)
    correct += (predicted[:-1] == labels).sum()
print('Test Accuracy', correct/total)

# TODO: print out some samples to make sure they make sense
# TODO: better loss measurements (top 20 precision, perplexity)

model.eval()
with open("lstm_predictions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['id','word'])
    for i, l in enumerate(open("input.txt"),1):
        words = [TEXT.vocab.stoi[word] for word in l.split(' ')]
        words = Variable(torch.Tensor(words).unsqueeze(1))
        out = model(words)
        predicted = out.numpy().argmax()
        writer.writerow([i,predicted])








#####################
# HIGHLY DUBIOUS
# Requires a different way of handling hidden, because there's a hidden for every vertical layer.
# I manually use multiple layers for the sake of dropout

hidden_size = 20
class kLSTM(nn.Module):# 128, 128, 20, 20, 2
    def __init__(self):
        super(kLSTM, self).__init__()
        self.embedding = nn.Embedding(word2vec.size(0),word2vec.size(1))
        self.embeddings.weight.data.copy_(word2vec)
        self.drop0 = nn.Dropout(dropout_rate)
        self.lstm1 = nn.LSTM(word2vec.size(1), hidden_size, 1) # 3rd argument is n_layers
        self.drop1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, 1)
        self.drop2 = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, len(TEXT.vocab))
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden1, hidden2): 
        # input is (batch size, sentence length) bs,n
        # hidden1 is ((n_layers,bs,hidden_size),(n_layers,bs,hidden_size))
        # embed the input integers
        embeds = self.embedding(input) # bs,n,300
        # put the batch along the second dimension
        embeds = embeds.transpose(0, 1) # n,bs,300
        out = self.drop0(embeds)
        out, hidden1 = self.lstm1(out, hidden1)
        out = self.drop1(out)
        out, hidden2 = self.lstm2(out, hidden2)
        out = self.drop2(out)
        # apply the linear and the softmax
        out = self.softmax(self.linear(out)) # n,bs,|V|
        return out, hidden1, hidden2

