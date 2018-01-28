# Text text processing library and methods for pretrained word embeddings
import torchtext
import torch
import numpy as np
from torchtext.vocab import Vectors, GloVe

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
    (train, val, test), batch_size=10, device=-1)

############################################

# Hyperparams
alpha = 0 # smoothing

p = np.zeros(len(TEXT.vocab)) + alpha
q = np.zeros(len(TEXT.vocab)) + alpha
ngood = 0
nbad = 0

for batch in train_iter:
	for i in range(batch.text.size()[1]):
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

def predict(text):
	ys = []
	for i in range(text.size()[1]):
		x = text.data.numpy()[:,i]
		sparse_x = np.zeros(len(TEXT.vocab))
		for word in x:
			sparse_x[word] = 1
		y = np.dot(r,sparse_x) + b
		if y > 0:
			ys.append(1)
		else:
			ys.append(2)
	return torch.Tensor(ys)

# write something to calculate the accuracy in-house