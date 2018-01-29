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
		self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
		self.embeddings.weight = TEXT.vocab.vectors
		self.linear = nn.Linear(TEXT.vocab.vectors.size(1), 2)

	def forward(self, inputs): # inputs (bs,words/sentence) 10,7
		bsz = inputs.size(0) # batch size might change
		embeds = self.embeddings(inputs) # 10,7,300
		out = out.sum(dim=1) # 10,300 (sum together embeddings across sentences)
		out = self.linear(x) # 300,2
		return out

model = CBOWLogReg(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
	ctr = 0
	for batch in train_iter:
		sentences = batch.text.transpose(1,0)
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
	bsz = batch.text.size(1) # batch size might change
	sentences = batch.text.transpose(1,0)
	labels = (batch.label==1).type(torch.LongTensor).data
	# change labels from 1,2 to 1,0
	outputs = model(sentences)
	_, predicted = torch.max(outputs.data, 1)
	total += labels.size(0)
	correct += (predicted == labels).sum()

print('val accuracy', correct/total)

torch.save(model.state_dict(), 'cbow.pkl')