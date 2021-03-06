# Text processing library and methods for pretrained word embeddings
import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torchtext.vocab import Vectors, GloVe
import csv

# Hyperparams
learning_rate = 0.001
bs = 10
num_epochs = 250

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

losses = []

for epoch in range(num_epochs):
	train_iter.init_epoch()
	ctr = 0
	for batch in train_iter:
		# TODO: is there a better way to sparsify?
		sentences = torch.zeros(bs,input_size)
		for i in range(batch.text.size(1)):
			x = batch.text.data.numpy()[:,i]
			for word in x:
				sentences[i,word] = 1 # += 1
		sentences = Variable(sentences)
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
		losses.append(loss.data[0])

np.save("../../models/lr_losses",np.array(losses))
torch.save(model.state_dict(), '../../models/lr.pkl')

# model.load_state_dict(torch.load('../../models/lr.pkl'))

correct = 0
total = 0
for batch in val_iter:
	bsz = batch.text.size(1) # batch size might change
	sentences = torch.zeros(bsz,input_size)
	for i in range(bsz):
		x = batch.text.data.numpy()[:,i]
		for word in x:
			sentences[i,word] = 1 # += 1
	sentences = Variable(sentences)
	labels = (batch.label==1).type(torch.LongTensor).data
	# change labels from 1,2 to 1,0
	outputs = model(sentences)
	_, predicted = torch.max(outputs.data, 1)
	total += labels.size(0)
	correct += (predicted == labels).sum()

print('val accuracy', correct/total)

def test(model):
	"All models should be able to be run with following command."
	upload = []
	# Update: for kaggle the bucket iterator needs to have batch_size 10
	# test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
	for batch in test_iter:
		# Your prediction data here (don't cheat!)
		bsz = batch.text.size(1) # batch size might change
		sentences = torch.zeros(bsz,input_size)
		for i in range(bsz):
			x = batch.text.data.numpy()[:,i]
			for word in x:
				sentences[i,word] = 1 # += 1
		sentences = Variable(sentences)
		probs = model(sentences)
		_, argmax = probs.max(1)
		upload += list(argmax.data)

	with open("lr_predictions.csv", "w") as f:
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