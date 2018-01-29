# Text processing library and methods for pretrained word embeddings
import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torchtext.vocab import Vectors, GloVe

# Hyperparams
filter_window = 3
n_featmaps = 100
bs = 10
dropout_rate = 0.5
num_epochs = 1
learning_rate = 0.001

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

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.embeddings = nn.Embedding(TEXT.vocab.vectors.size()[0],TEXT.vocab.vectors.size()[1])
		self.embeddings.weight.data = TEXT.vocab.vectors
		self.conv = nn.Conv2d(1,n_featmaps,kernel_size=(filter_window,300))
		self.maxpool = nn.AdaptiveMaxPool1d(1)
		self.linear = nn.Linear(n_featmaps, 2)
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, inputs): # inputs (bs,words/sentence) 10,7
		bsz = inputs.size()[0] # batch size might change
		embeds = self.embeddings(inputs) # 10,7,300
		out = embeds.unsqueeze(1) # 10,1,7,300
		out = F.relu(self.conv(out)) # 10,100,6,1
		out = out.view(bsz,n_featmaps,-1) # 10,100,6
		out = self.maxpool(out) # 10,100,1
		out = out.view(bsz,-1) # 10,100
		out = self.linear(out) # 10,2
		out = self.dropout(out) # 10,2
		return out
# TODO: what about weight constraints?

model = CNN()
criterion = nn.CrossEntropyLoss() # accounts for the softmax component?
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
		nn.utils.clip_grad_norm(model.parameters(), 3)
		ctr += 1
		if ctr % 100 == 0:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
				%(epoch+1, num_epochs, ctr, len(train)//bs, loss.data[0]))

model.eval() # lets dropout layer know that this is the test set
correct = 0
total = 0
for batch in val_iter:
	sentences = batch.text.transpose(1,0)
	labels = (batch.label==1).type(torch.LongTensor).data
	# change labels from 1,2 to 1,0
	outputs = model(sentences)
	_, predicted = torch.max(outputs.data, 1)
	total += labels.size(0)
	correct += (predicted == labels).sum()

print('test accuracy', correct/total)

torch.save(model.state_dict(), 'cnn.pkl')