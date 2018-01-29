# Text processing library and methods for pretrained word embeddings
import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchtext.vocab import Vectors, GloVe

# Hyperparams
learning_rate = 0.001
bs = 10
num_epochs = 1

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

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

print("Word embeddings size ", TEXT.vocab.vectors.size())

############################################
# With help from Yunjey Pytorch Tutorial on Github

# Hyperparams
input_size = len(TEXT.vocab)

class CBOWLogReg(nn.Module):
	def __init__(self, input_size):
		super(CBOWLogReg, self).__init__()
		self.embeddings = nn.Embedding(TEXT.vocab.vectors.size())
		self.embeddings.weight = TEXT.vocab.vectors

		self.linear = # ??????????????????????????????

	def forward(self, x):
		out = self.linear(x)
		return out

model = CBOWLogReg(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# TODO: apparently the ce loss takes care of the softmax so I don't have to
# but I don't really understand how it works

for epoch in range(num_epochs):
	ctr = 0
	for batch in train_iter:
		# TODO: is there a better way to sparsify?
		sentences = Variable(torch.zeros(bs,input_size))
		for i in range(batch.text.size()[1]):
			x = batch.text.data.numpy()[:,i]
			for word in x:
				sentences[i,word] = 1 # += 1
		labels = (batch.label==1).type(torch.LongTensor)
		# change labels from 1,2 to 1,0
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
	bsz = batch.text.size()[1] # batch size might change
	sentences = Variable(torch.zeros(bsz,input_size))
	for i in range(bsz):
		x = batch.text.data.numpy()[:,i]
		for word in x:
			sentences[i,word] = 1 # += 1
	labels = (batch.label==1).type(torch.LongTensor).data
	# change labels from 1,2 to 1,0
	outputs = model(sentences)
	_, predicted = torch.max(outputs.data, 1)
	total += labels.size(0)
	correct += (predicted == labels).sum()

print('test accuracy', correct/total)