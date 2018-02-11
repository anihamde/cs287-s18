import numpy as np
from collections import Counter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv

# Hyperparameters
bs = 10 # batch size
n = 5 # receptive field
hidden_size = 100
learning_rate = .001
weight_decay = 0
num_epochs = 10

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
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) # feel free to alter path
print("Word embeddings size ", TEXT.vocab.vectors.size())
print("REMINDER!!! Did you create ../../models/HW2?????")

# TODO: mixture of models with interpolated trigram (fixed or learned weights)
# TODO: weight decay penalty (excluding bias terms) (want to try weight clippings too?)
# TODO: bengio extensions
# TODO: parallel computation (make everything cuda)
# TODO: our own extensions (multichannel with glove, static/dynamic, etc?) (conv layers) (dropout) (recurrence) (pad at the beginning?)

class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
        self.embeddings.weight.data = TEXT.vocab.vectors
        self.h = nn.Linear(n*300,hidden_size)
        self.u = nn.Linear(hidden_size,len(TEXT.vocab))
        self.w = nn.Linear(n*300,len(TEXT.vocab))
    def forward(self, inputs): # inputs (batch size, "sentence" length) bs,n
        embeds = self.embeddings(inputs) # bs,n,300
        embeds = embeds.view(-1,n*300) # bs,n*300
        out = F.tanh(self.h(embeds)) # bs,hidden_size
        out = self.u(out) # bs,|V|
        out += self.w(embeds) # bs,|V|
        out = F.softmax(out,dim=1)
        return out


model = NNLM()
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
params = filter(lambda x: x.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

losses = []
model.train()
for i in range(num_epochs):
    ctr = 0
    for batch in iter(train_iter):
        sentences = batch.text.transpose(1,0).cuda() # bs,n
        if sentences.size(1) < n+1: # make sure sentence length is long enough
            pads = Variable(torch.zeros(sentences.size(0),n+1-sentences.size(1))).type(torch.cuda.LongTensor)
            sentences = torch.cat([pads,sentences],dim=1)
        for j in range(n,sentences.size(1)):
            out = model(sentences[:,j-n:j])
            loss = criterion(out,sentences[:,j])
            model.zero_grad()
            loss.backward()
            optimizer.step()
        ctr += 1
        if ctr % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                %(i+1, num_epochs, ctr, len(train_iter), loss.data[0]))
        losses.append(loss.data[0])

    # can add a net_flag to these file names. and feel free to change the paths
    np.save("../../models/HW2/nnlm_losses.npy",np.array(losses))
    torch.save(model.state_dict(), '../../models/HW2/nnlm.pkl')

# model.load_state_dict(torch.load('../../models/HW2/nnlm.pkl'))

model.eval()
correct = total = 0
precisionmat = (1/np.arange(1,21))[::-1].cumsum()[::-1]
precisionmat = torch.FloatTensor(precisionmat.copy()) # hm
precision = 0
crossentropy = 0

for batch in iter(val_iter):
    sentences = batch.text.transpose(1,0).cuda() # bs, n
    if sentences.size(1) < n+1: # make sure sentence length is long enough
        pads = Variable(torch.zeros(sentences.size(0),n+1-sentences.size(1))).type(torch.cuda.LongTensor)
        sentences = torch.cat([pads,sentences],dim=1)
    for j in range(n,sentences.size(1)):
        # precision
        out = model(sentences[:,j-n:j]) # bs,|V|
        labels = sentences[:,j] # bs
        _, outsort = torch.sort(out,dim=1)
        outsort = outsort[:,:20]
        inds = (outsort-labels.unsqueeze(1)==0)
        inds = inds.sum(dim=0).data.type(torch.FloatTensor)
        precision += inds.dot(precisionmat)
        # cross entropy
        crossentropy += F.cross_entropy(out,labels)
        # plain ol accuracy
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted==labels.data).sum()
        # if total % 500 == 0:
        # DEBUGGING: see the rest in trigram.py
        print('we are on example', total)
        for s in range(bs): # when you test this, make all the += into =
            print([TEXT.vocab.itos[w] for w in sentences[s,j-2:j]])
            print([TEXT.vocab.itos[w] for w in outsort[s]])
        print('Test Accuracy', correct/total)
        print('Precision',precision/total)
        print('Perplexity',torch.exp(bs*crossentropy/total).data[0])

print('Test Accuracy', correct/total)
print('Precision',precision/total)
print('Perplexity',torch.exp(bs*crossentropy/total).data[0])
# F.cross_entropy averages instead of adding

model.eval()
with open("nnlm_predictions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['id','word'])
    for i, l in enumerate(open("input.txt"),1):
        words = [TEXT.vocab.stoi[word] for word in l.split(' ')]
        words = Variable(torch.cuda.LongTensor(words[-1-n:-1]))
        out = model(words)
        _, predicted = torch.sort(out,dim=1)
        predicted = predicted[0,:20].data.tolist()
        writer.writerow([i,' '.join(predicted)])