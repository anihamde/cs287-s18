import numpy as np
from collections import Counter

# Hyperparameters
bs = 10 # batch size
alpha_t = .4 # trigram probability
alpha_b = .2 # bigram probability

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

it = iter(train_iter)
batch = next(it) 
print("Size of text batch [max bptt length, batch size]", batch.text.size())
print("Second in batch", batch.text[:, 2])
print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 2].data]))
batch = next(it)
print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 2].data]))

uni = Counter()
bi = Counter()
tri = Counter()
biprev = [1] * bs # this uses padding
triprev = [1] * bs * 2

for b in iter(train_iter):
    txt = b.text.data
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

# TODO: experiment with ignoring EOS for unigrams, like this
# uni[TEXT.vocab.stoi["<eos>"]] = 0

unisum = sum(uni) # just normalize once for unigrams
for k in uni:
    uni[k] *= (1 - alpha_b - alpha_t) / unisum

def predict(l):
    # filter
    bifilt = Counter({k:bi[k] for k in bi if k[0]==l[-1]})
    trifilt = Counter({k:tri[k] for k in tri if k[0]==l[-2] and k[1]==l[-1]})
    # normalize
    bisum = sum(bifilt.values())
    trisum = sum(trifilt.values())
    # combine
    total = uni
    for k in bifilt:
        total[k[-1]] += bifilt[k] * alpha_b / bisum
    for k in trifilt:
        total[k[-1]] += trifilt[k] * alpha_t / trisum
    # select top results
    return [TEXT.vocab.itos[i] for i,c in total.most_common(20)]

# TODO: write an in-house evaluator, with perplexity or precision metric

with open("sample.txt", "w") as fout: 
    print("id,word", file=fout)
    for i, l in enumerate(open("input.txt"), 1):
        words = l.split(' ')[:-1]
        words = [TEXT.vocab.stoi[word] for word in words]
        print("%d,%s"%(i, " ".join(predict(words))), file=fout)
