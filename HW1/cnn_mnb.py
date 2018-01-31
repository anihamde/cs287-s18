# Text processing library and methods for pretrained word embeddings
import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchtext.vocab import Vectors, GloVe
import csv

# Hyperparams
filter_window = 3
n_featmaps = 100
bs = 10
dropout_rate = 0.5
num_epochs = 15 # from 30
learning_rate = 0.001
constraint = 3

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
    (train, val, test), batch_size=bs, device=-1, repeat=False)

# glove = GloVe(name='6B',dim=300)
TEXT.vocab.load_vectors(vectors=Vectors('glove.6B.300d.txt'))
glove = TEXT.vocab.vectors

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

print("Word embeddings size ", TEXT.vocab.vectors.size())
word2vec = TEXT.vocab.vectors

############################################

# Hyperparams
alpha = 1 # smoothing

p = np.zeros(len(TEXT.vocab)) + alpha
q = np.zeros(len(TEXT.vocab)) + alpha
ngood = 0
nbad = 0

for batch in train_iter:
    for i in range(batch.text.size(1)):
        x = batch.text.data.numpy()[:,i]
        y = batch.label.data.numpy()[i]
        sparse_x = np.zeros(len(TEXT.vocab))
        for word in x:
            sparse_x[word] = 1 # += 1
        if y == 1:
            p += sparse_x
            ngood += 1
        elif y == 2:
            q += sparse_x
            nbad += 1
        else:
            pass

r = np.log((p/np.linalg.norm(p))/(q/np.linalg.norm(q)))
b = np.log(ngood/nbad)

print('mnb bias is', b)

############################################
# With help from Yunjey Pytorch Tutorial on Github

class CNN(nn.Module):
  def __init__(self):
      super(CNN, self).__init__()
      self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
      self.embeddings.weight.data.copy_(word2vec)
      self.conv = nn.Conv2d(1,n_featmaps,kernel_size=(filter_window,300))
      self.maxpool = nn.AdaptiveMaxPool1d(1)
      self.linear = nn.Linear(n_featmaps, 2)
      self.dropout = nn.Dropout(dropout_rate)

  def forward(self, inputs, ys): # inputs (bs,words/sentence) 10,7
      bsz = inputs.size(0) # batch size might change
      if inputs.size(1) < 3: # padding issues on really short sentences
          pads = Variable(torch.zeros(bsz,3-inputs.size(1)))
          inputs = torch.cat([inputs,pads],dim=1)
      embeds = self.embeddings(inputs) # 10,7,300
      out = embeds.unsqueeze(1) # 10,1,7,300
      out = F.relu(self.conv(out)) # 10,100,6,1
      out = out.view(bsz,n_featmaps,-1) # 10,100,6
      out = self.maxpool(out) # 10,100,1
      out = out.view(bsz,-1) # 10,100
      out = torch.cat([out,ys],dim=1) # 10,101
      out = self.linear(out) # 10,2
      out = self.dropout(out) # 10,2
      return out

model = CNN()
if torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss() # accounts for the softmax component?
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []

for epoch in range(num_epochs):
    train_iter.init_epoch()
    ctr = 0
    for batch in train_iter:
        ys = torch.zeros(batch.text.size(1)) # mnb outputs
        for i in range(batch.text.size(1)):
            x = batch.text.data.numpy()[:,i]
            y = batch.label.data.numpy()[i]
            sparse_x = np.zeros(len(TEXT.vocab))
            for word in x:
                sparse_x[word] = 1 # += 1
            ys[i] = (np.dot(r,sparse_x) + b)>0
        sentences = batch.text.transpose(1,0)
        labels = (batch.label==1).type(torch.LongTensor)
        # change labels from 1,2 to 1,0
        sentences = sentences.cuda()
        ys = ys.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(sentences, ys)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        nn.utils.clip_grad_norm(model.parameters(), constraint)
        ctr += 1
        if ctr % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                %(epoch+1, num_epochs, ctr, len(train)//bs, loss.data[0]))
        losses.append(loss.data[0])

    # normally you'd tab these back a level, but i'm paranoid
    np.save("../../models/cnn_mnb_losses",np.array(losses))
    torch.save(model.state_dict(), '../../models/cnn_mnb.pkl')

# model.load_state_dict(torch.load('../../models/0cnn.pkl'))

model.eval() # lets dropout layer know that this is the test set
correct = 0
total = 0
for batch in val_iter:
    ys = torch.zeros(batch.text.size(1)) # mnb outputs
    for i in range(batch.text.size(1)):
        x = batch.text.data.numpy()[:,i]
        y = batch.label.data.numpy()[i]
        sparse_x = np.zeros(len(TEXT.vocab))
        for word in x:
            sparse_x[word] = 1 # += 1
        ys[i] = (np.dot(r,sparse_x) + b)>0
    sentences = batch.text.transpose(1,0)
    sentences = sentences.cuda()
    ys = ys.cuda()
    labels = labels.cuda()
    labels = (batch.label==1).type(torch.LongTensor).data
    # change labels from 1,2 to 1,0
    outputs = model(sentences,ys)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('test accuracy', correct/total)

def test(model):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    # test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        ys = torch.zeros(batch.text.size(1)) # mnb outputs
        for i in range(batch.text.size(1)):
            x = batch.text.data.numpy()[:,i]
            y = batch.label.data.numpy()[i]
            sparse_x = np.zeros(len(TEXT.vocab))
            for word in x:
                sparse_x[word] = 1 # += 1
            ys[i] = (np.dot(r,sparse_x) + b)>0
        sentences = batch.text.transpose(1,0)
        sentences = sentences.cuda()
        ys = ys.cuda()
        labels = labels.cuda()
        probs = model(sentences,ys)
        _, argmax = probs.max(1)
        upload += list(argmax.data)

    with open("cnn_mnb_predictions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Id','Cat'])
        idcntr = 0
        for u in upload:
            if u == 0:
                u = 2
            writer.writerow([idcntr,u])
            idcntr += 1
            # f.write(str(u) + "\n")

test(model)
