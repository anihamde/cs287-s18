import numpy as np
from collections import Counter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv

# Hyperparameters
bs = 10 # batch size
n = 10 # receptive field
hidden_size = 100
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
criterion = nn.CrossEntropyLoss()
params = filter(lambda x: x.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

losses = []
model.train()
for i in range(num_epochs):
    print(i)
    train_iter.init_epoch()
    ctr = 0
    for batch in iter(train_iter):
        sentences = batch.text.transpose(1,0) # bs,n
        if sentences.size(1) < n+1: # make sure sentence length is long enough
            pads = Variable(torch.zeros(sentences.size(0),n+1-sentences.size(1))).type(torch.LongTensor)
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
                %(epoch+1, num_epochs, ctr, len(train_iter), loss.data[0]))
        losses.append(loss.data[0])

    # can add a net_flag to these file names. and feel free to change the paths
    np.save("../../models/HW2/nnlm_losses.npy",np.array(losses))
    torch.save(model.state_dict(), '../../models/HW2/nnlm.pkl')

# model.load_state_dict(torch.load('../../models/HW2/nnlm.pkl'))

model.eval()
correct = total = 0
precisionmat = []

for i in range(0,20):
    precisionmat.append(1.0/(1.0+i))

for i in range(0,20):
    precisionmat[i] = sum(precisionmat[i:20])

precisioncalc = 0
precisioncntr = 0
crossentropy = 0

for batch in iter(val_iter):
    sentences = batch.text.transpose(1,0) # bs, n
    if sentences.size(1) < n+1: # make sure sentence length is long enough
        pads = Variable(torch.zeros(sentences.size(0),n+1-sentences.size(1))).type(torch.LongTensor)
        sentences = torch.cat([pads,sentences],dim=1)
    for j in range(n,sentences.size(1)):
        # precision
        out = model(sentences[:,j-n:j])
        _,indices = torch.sort(out,descending=True)
        indices20 = indices[:,0:20]
        labels = sentences[:,j]
        labels2 = labels.unsqueeze(0)
        labels2 = labels2.permute(1,0)
        indicmat = np.where((indices20 - labels2.expand(labels2.size()[0],20)).data.numpy() == 0)
        for k in range(0,len(indicmat[0])):
            colm = indicmat[1][k]
            precisioncalc += precisionmat[colm]
        precisioncntr += len(labels)
        # cross entropy
        crossentropy += F.cross_entropy(out,labels)
        # plain ol accuracy
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted.numpy() == labels.data.numpy()).sum()


print('Test Accuracy', correct/total)
print('Precision',precisioncalc/(20*precisioncntr))
print('Perplexity',torch.exp(crossentropy/precisioncntr).data.numpy())

model.eval()
with open("nnlm_predictions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['id','word'])
    for i, l in enumerate(open("input.txt"),1):
        words = [TEXT.vocab.stoi[word] for word in l.split(' ')]
        words = Variable(torch.LongTensor(words[-1-n:-1]))
        out = model(words)
        predicted = out.data.numpy()
        predicted = predicted.argsort()
        predicted = predicted[:,:20]
        writer.writerow([i,predicted])