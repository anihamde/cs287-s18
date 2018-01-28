# Text processing library and methods for pretrained word embeddings
import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torchtext.vocab import Vectors, GloVe

# Hyperparams
learning_rate = 0.001
bs = 10

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
	(train, val, test), batch_size=bs, device=-1)

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

print("Word embeddings size ", TEXT.vocab.vectors.size())

############################################
# With help from Yunjey Pytorch Tutorial on Github

# Hyperparams
input_size = len(TEXT.vocab)

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 2)
    
    def forward(self, x):
        out = self.linear(x)
        return out

model = LogisticRegression(input_size)
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

ctr = 0
for epoch in range(num_epochs):
	for batch in train_iter:
		sentences = Variable(torch.zeros(bs,input_size))
		for i in range(batch.text.size()[1]):
			x = batch.text.data.numpy()[:,i]
			for word in x:
				sentences[i,word] = 1 # += 1
		labels = batch.label
		# TODO: change labels from "1" and "2"
		optimizer.zero_grad()
		outputs = model(sentences)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		ctr += 1
		if ctr % 100 == 0:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
				%(epoch+1, num_epochs, ctr, len(train)//bs, loss.data[0]))

correct = 0
total = 0
for batch in val_iter:
	sentences = Variable(torch.zeros(bs,input_size))
		for i in range(batch.text.size()[1]):
			x = batch.text.data.numpy()[:,i]
			for word in x:
				sentences[i,word] = 1 # += 1
	labels = batch.label
	# TODO: change labels from "1" and "2"
	outputs = model(sentences)
	_, predicted = torch.max(outputs.data, 1)
	total += labels.size(0)
	correct += (predicted == labels).sum()
