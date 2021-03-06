import time
import numpy as np
from collections import Counter
import csv
import copy
import argparse

timenow = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size','-bs',type=int,default=10,help='set training batch size. default = 10.')
parser.add_argument('--alphab','-ab',type=float, default=0.4)
parser.add_argument('--alphat','-at',type=float, default=0.25)
parser.add_argument('--epsilon','-e',type=float,default=0.01)
args = parser.parse_args()

# Hyperparameters
bs = args.batch_size
alpha_b = args.alphab
alpha_t = args.alphat
eps = args.epsilon

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

# TODO: use argparse. and have an option to use train.5k
if False:
    TEXT.build_vocab(train, max_size=1000)
    len(TEXT.vocab)

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=bs, device=-1, bptt_len=32, repeat=False)

# it = iter(train_iter)
# batch = next(it) 
# print("Size of text batch [max bptt length, batch size]", batch.text.size())
# print("Second in batch", batch.text[:, 2])
# print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 2].data]))
# batch = next(it)
# print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 2].data]))

# TODO: tiny epsilon normalization to stop entropy from infinity-- is that ok? when a word has prob 0, we do eps CE
# otherwise, how to stop entropy from infinity?
# TODO: weights as a function of frequency of prev 2 words (bengio)

# Maybe I can come up with something more efficient than a counter?
uni = Counter()
bi = Counter()
tri = Counter()
biprev = [1] * bs # "1" is <pad>
triprev = [1] * bs * 2

for batch in iter(train_iter):
    txt = batch.text.data
    uni.update(txt.view(-1).tolist()) # throw all words into bag
    bi0 = biprev + txt[:-1,:].view(-1).tolist()
    bi1 = txt.view(-1).tolist()
    biprev = txt[-1,:].view(-1).tolist()
    bi.update(zip(bi0,bi1))
    tri0 = triprev + txt[:-2,:].view(-1).tolist()
    tri1 = triprev[bs:] + txt[:-1,:].view(-1).tolist()
    tri2 = txt.view(-1).tolist()
    triprev = txt[-2:,:].view(-1).tolist()
    tri.update(zip(tri0,tri1,tri2))

print("Done training!")
# TODO: experiment with ignoring EOS for unigrams, like this
# uni[TEXT.vocab.stoi["<eos>"]] = 0

unisum = sum(uni.values()) # just normalize once for unigrams
for k in uni:
    uni[k] *= (1 - alpha_b - alpha_t) / unisum
print("Done normalizing unigram counter!")

def predict(l):
    # filter
    bifilt = Counter({k:bi[k] for k in bi if k[0]==l[-1]})
    trifilt = Counter({k:tri[k] for k in tri if k[0]==l[-2] and k[1]==l[-1]})
    # normalize
    bisum = sum(bifilt.values())
    trisum = sum(trifilt.values())
    # combine
    total = copy.copy(uni) # shallow copy
    for k in bifilt:
        total[k[-1]] += bifilt[k] * alpha_b / bisum
    for k in trifilt:
        total[k[-1]] += trifilt[k] * alpha_t / trisum
    return total

# enum_ctr = 0
# with open("trigram_predictions.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(['id','word'])
#     for i, l in enumerate(open("input.txt"),1):
#         words = l.split(' ')[:-1]
#         words = [TEXT.vocab.stoi[word] for word in words]
#         out = predict(words)
#         out = [TEXT.vocab.itos[i] for i,c in out.most_common(20)]
#         writer.writerow([i,' '.join(out)])
#         enum_ctr += 1
#         if enum_ctr % 100 == 0:
#             print(enum_ctr)
# print("Done writing kaggle text!")

# Evaluator
correct = total = 0
precisionmat = (1/np.arange(1,21))[::-1].cumsum()[::-1]
precision = 0
crossentropy = 0

for batch in iter(val_iter):
    sentences = batch.text.data.transpose(1,0) # bs,n
    if sentences.size(1) < 3: # make sure sentence length is long enough
        pads = torch.zeros(sentences.size(0),3-sentences.size(1)).type(torch.LongTensor)
        sentences = torch.cat([pads,sentences],dim=1)
    for sentence in sentences:
        for j in range(2,sentence.size(0),5): # added that 5 for spice/variety
            # precision
            out = predict(sentence[j-2:j])
            indices = [a for a,b in out.most_common(20)]
            label = sentence[j]
            if label in indices:
                precision += precisionmat[indices.index(label)]
            # cross entropy
            prob = (1-eps)*out[label]+(eps/len(TEXT.vocab))
            crossentropy -= np.log(prob)
            # plain ol accuracy
            total += 1
            correct += (indices[0] == label)
            # if total % 500 == 0: print stats
    if total>1000: # that's enough
        break

# DEBUGGING:
# print('we are on example', total)
# print([TEXT.vocab.itos[w] for w in sentence[j-2:j]])
# print(TEXT.vocab.itos[label])
# print([TEXT.vocab.itos[w] for w in indices])
# print(1/out[label])
# print(precisionmat[indices.index(label)] if label in indices else 0)
# print(indices[0] == label)

    print('Test Accuracy', correct/total)
    print('Precision',precision/total)
    print('Perplexity',np.exp(crossentropy/total))

# print(time.time()-timenow)
