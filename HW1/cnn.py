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
num_epochs = 75
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

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

print("Word embeddings size ", TEXT.vocab.vectors.size())

############################################
# With help from Yunjey Pytorch Tutorial on Github

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
		self.embeddings.weight.data = TEXT.vocab.vectors
		self.conv = nn.Conv2d(1,n_featmaps,kernel_size=(filter_window,300))
		self.maxpool = nn.AdaptiveMaxPool1d(1)
		self.linear = nn.Linear(n_featmaps, 2)
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, inputs): # inputs (bs,words/sentence) 10,7
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
		out = self.linear(out) # 10,2
		out = self.dropout(out) # 10,2
		return out

model = CNN()
criterion = nn.CrossEntropyLoss() # accounts for the softmax component?
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []

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
		nn.utils.clip_grad_norm(model.parameters(), constraint)
		ctr += 1
		if ctr % 100 == 0:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
				%(epoch+1, num_epochs, ctr, len(train)//bs, loss.data[0]))
		losses.append(loss)


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

torch.save(model.state_dict(), '../../models/cnn.pkl')

def test(model):
	"All models should be able to be run with following command."
	upload = []
	# Update: for kaggle the bucket iterator needs to have batch_size 10
	# test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
	for batch in test_iter:
		# Your prediction data here (don't cheat!)
		sentences = batch.text.transpose(1,0)
		probs = model(sentences)
		_, argmax = probs.max(1)
		upload += list(argmax.data)

	with open("cnn_predictions.csv", "w") as f:
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

np.save("cnn_losses",np.array(losses))





###############################3

# Multichannel CNN
class MCNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
		self.embeddings.weight.data = TEXT.vocab.vectors
		self.s_embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
		self.s_embeddings.weight.data = TEXT.vocab.vectors # make these embeddings something else???
		self.s_embeddings.weight.requires_grad = False
		self.conv = nn.Conv2d(2,n_featmaps,kernel_size=(filter_window,300))
		self.maxpool = nn.AdaptiveMaxPool1d(1)
		self.linear = nn.Linear(n_featmaps, 2)
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, inputs): # inputs (bs,words/sentence) 10,7
		bsz = inputs.size(0) # batch size might change
		if inputs.size(1) < 3: # padding issues on really short sentences
			pads = Variable(torch.zeros(bsz,3-inputs.size(1)))
			inputs = torch.cat([inputs,pads],dim=1)
		embeds = self.embeddings(inputs) # 10,7,300
		embeds = embeds.unsqueeze(1) # 10,1,7,300
		s_embeds = self.s_embeddings(inputs) # 10,7,300
		s_embeds = s_embeds.unsqueeze(1) # 10,1,7,300
		out = torch.cat([embeds,s_embeds],dim=1) # 10,2,7,300
		out = F.relu(self.conv(out)) # 10,100,6,1
		out = out.view(bsz,n_featmaps,-1) # 10,100,6
		out = self.maxpool(out) # 10,100,1
		out = out.view(bsz,-1) # 10,100
		out = self.linear(out) # 10,2
		out = self.dropout(out) # 10,2
		return out

model = MCNN()
criterion = nn.CrossEntropyLoss() # accounts for the softmax component?
params = filter(lambda x: x.requires_grad, models.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)