# Text text processing library and methods for pretrained word embeddings
import torchtext
import torch
import numpy as np
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable
import csv

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
	(train, val, test), batch_size=10, device=-1, repeat=False)

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

# model takes in a batch.text and return a bs*2 tensor of probs
def predict(text):
	ys = torch.zeros(text.size(1),2)
	for i in range(text.size(1)):
		x = text.data.numpy()[:,i]
		sparse_x = np.zeros(len(TEXT.vocab))
		for word in x:
			sparse_x[word] = 1
		y = np.dot(r,sparse_x) + b
		if y > 0:
			ys[i,1] = 1
		else:
			ys[i,0] = 1
	return Variable(ys)

def inhousepredict(batch):
	ys = torch.zeros(batch.text.size()[1])
	labs = batch.label
	for i in range(batch.text.size()[1]):
		x = batch.text.data.numpy()[:,i]
		sparse_x = np.zeros(len(TEXT.vocab))
		for word in x:
			sparse_x[word] = 1
		y = np.dot(r,sparse_x) + b
		if y > 0:
			ys[i] = 1

	check = (ys.numpy() - labs.data.numpy())%2

	incorrect = sum(check)

	return incorrect, len(check)

def test(model):
	"All models should be able to be run with following command."
	upload = []
	# Update: for kaggle the bucket iterator needs to have batch_size 10
	# test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
	for batch in test_iter:
		# Your prediction data here (don't cheat!)
		probs = model(batch.text)
		_, argmax = probs.max(1)
		upload += list(argmax.data)

	with open("mnb_predictions.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerow(['Id','Cat'])
		idcntr = 0
		for u in upload:
			if u == 0:
				u = 2
			writer.writerow([idcntr,u])
			idcntr += 1
			# f.write(str(u) + "\n")

test(predict)
# incorrect = 0
# total = 0

# for batch in val_iter:
# 	inter_vec = inhousepredict(batch)
# 	incorrect += inter_vec[0]
# 	total += inter_vec[1]

# print((1-incorrect/total)*100)

# realizing that TEXT.label uses 2 and 1, not 0 and 1