import time
import numpy as np
from collections import Counter

timenow = time.time()

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

print("Training done!")
# TODO: experiment with ignoring EOS for unigrams, like this
# uni[TEXT.vocab.stoi["<eos>"]] = 0

unisum = sum(uni.values()) # just normalize once for unigrams
for k in uni:
    uni[k] *= (1 - alpha_b - alpha_t) / unisum

print("Done with normalizing!")

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
    return total
    # return [TEXT.vocab.itos[i] for i,c in total.most_common(20)]

print("Defined predict")


enum_cntr = 0

with open("sample.txt", "w") as fout: 
    print("id,word", file=fout)
    for i, l in enumerate(open("input.txt"), 1):
        words = l.split(' ')[:-1]
        words = [TEXT.vocab.stoi[word] for word in words]
        total = predict(words)
        print("%d,%s"%(i, " ".join([TEXT.vocab.itos[i] for i,c in total.most_common(20)])), file=fout)
        enum_cntr += 1

        if enum_cntr % 100 == 0:
            print(enum_cntr)

print("Done with making predictions!")

# Evaluator

n = 2
correct = total = 0
precisionmat = 1/(range(1,21))

for i in range(0,20):
    precisionmat[i] = sum(precisionmat[i:20])

precisioncalc = 0
precisioncntr = 0
crossentropy = 0

print("Beginning validation!")

for batch in iter(val_iter):
    sentences = batch.text.transpose(1,0)
    if sentences.size(1) < n+1: # make sure sentence length is long enough
        pads = Variable(torch.zeros(sentences.size(0),n+1-sentences.size(1))).type(torch.LongTensor)
        sentences = torch.cat([pads,sentences],dim=1)
    for sentence in sentences:
        for j in range(n,sentence.size(0)):
            # precision
            words = l.split(' ')[:-1]
            sentence = [TEXT.vocab.stoi[word] for word in words]

            out = predict(sentence) # TODO: Fix this
            # out = model(sentence[j-n:j])
            sorte,indices = torch.sort(out,desc=True)
            indices20 = indices[0:20]
            # _, predicted = torch.max(out.data, 1)
            label = sentence[j]
            
            indic = np.where(indices20 - label == 0)

            precisioncalc += precisionmat[indic]

            precisioncntr += 1

            # cross entropy
            crossentropy += F.cross_entropy(out,label)

            # plain ol accuracy
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum()

            if precisioncntr % 1000 == 0:
                print(precisioncntr)

print('Test Accuracy', correct/total)
print('Precision',precisioncalc/(20*precisioncntr))
print('Perplexity',torch.exp(crossentropy/precisioncntr))

print(time.time()-timenow)
