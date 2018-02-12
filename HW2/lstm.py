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
word2vec = TEXT.vocab.vectors
print("REMINDER!!! Did you create ../../models/HW2?????")

# TODO: learning rate decay? (zaremba has specific instructions for this)
# TODO: multichannel tests (with glove and stuff)
# TODO: bidirectional, gru
# TODO: replace dropouts with functional dropouts
# TODO: make hidden sizes bigger (500 according to piazza)
# TODO: early stopping
# TODO: minibatch size 20 (according to zaremba), and clip the grads normalized by minibatch size

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class dLSTM(nn.Module):
    def __init__(self):
        super(dLSTM, self).__init__()
        self.embedding = nn.Embedding(word2vec.size(0),word2vec.size(1))
        self.embedding.weight.data.copy_(word2vec)
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
    
    def initHidden(self):
        hidden = []
        for i in range(n_layers):
            hold = torch.zeros(n_layers, bs, hidden_size).type(torch.FloatTensor)
            if torch.cuda.is_available():
                hold = hold.cuda()
            hidden.append( Variable(hold) )
        return hidden

model = dLSTM()
if torch.cuda.is_available():
    model.cuda()
    print("CUDA is available, assigning to GPU.")
criterion = nn.NLLLoss()
params = filter(lambda x: x.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

losses = []
model.train()
for i in range(num_epochs):
    ctr = 0
    # initialize hidden vector
    hidden = model.initHidden()
    for batch in iter(train_iter):
        sentences = batch.text # Variable of LongTensor of size (n,bs)
        if torch.cuda.is_available():
            sentences = sentences.cuda()
        out, hidden = model(sentences, hidden)
        loss = 0
        for j in range(sentences.size(1)-1):
            loss += criterion(out[j], sentences[j+1])
        model.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm(params, constraint) # is this right?
        optimizer.step()
        # hidden vector is automatically saved for next batch
        ctr += 1
        losses.append(loss.data[0])
        if ctr % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                %(i+1, num_epochs, ctr, len(train_iter), sum(losses[-500:])/len(losses[-500:])  ))
        hidden = repackage_hidden(hidden)

    # can add a net_flag to these file names. and feel free to change the paths
    np.save("../../models/HW2/lstm_losses",np.array(losses))
    torch.save(model.state_dict(), '../../models/HW2/lstm.pkl')

# model.load_state_dict(torch.load('../../models/lstm.pkl'))

model.eval()
correct = total = 0
precisionmat = (1/np.arange(1,21))[::-1].cumsum()[::-1]
precisionmat = torch.FloatTensor(precisionmat.copy())
precision = 0
crossentropy = 0

hidden = model.initHidden()
for batch in iter(val_iter):
    sentences = batch.text
    if torch.cuda.is_available():
        sentences = sentences.cuda()
    out, hidden = model(sentences, hidden)
    for j in range(sentences.size(1)):
        out = out[j] # bs,|V|
        labels = sentences[j] # bs
        # cross entropy
        crossentropy += F.cross_entropy(out,labels)
        # precision
        out, labels = out.data, labels.data
        _, outsort = torch.sort(out,dim=1,descending=True)
        outsort = outsort[:,:20]
        inds = (outsort-labels.unsqueeze(1)==0)
        inds = inds.sum(dim=0).type(torch.FloatTensor)
        precision += inds.dot(precisionmat)
        # plain ol accuracy
        _, predicted = torch.max(out, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum()
        if total % 500 == 0:
            # DEBUGGING: see the rest in trigram.py
            print('we are on example', total)
            print('Test Accuracy', correct/total)
            print('Precision',precision/total)
            print('Perplexity',torch.exp(bs*crossentropy/total).data[0])

print('Test Accuracy', correct/total)
print('Precision',precision/total)
print('Perplexity',torch.exp(bs*crossentropy/total).data[0])
# F.cross_entropy averages instead of adding

model.eval()
with open("lstm_predictions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['id','word'])
    for i, l in enumerate(open("input.txt"),1):
        words = [TEXT.vocab.stoi[word] for word in l.split(' ')]
        words = Variable(torch.cuda.LongTensor(words).unsqueeze(1))
        hidden = (Variable(torch.zeros(n_layers, 1, hidden_size)).cuda(),
            Variable(torch.zeros(n_layers, 1, hidden_size)).cuda())
        out, _ = model(words,hidden)
        _, predicted = torch.sort(out,dim=1,descending=True)
        predicted = predicted[0,:20].data.tolist()
        writer.writerow([str(i),' '.join(predicted)])