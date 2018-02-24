import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions # this package provides a lot of nice abstractions for policy gradients
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
import spacy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import csv
import time
import math
import copy
import argparse

from models import AttnNetwork, CandList, S2S
from helpers import asMinutes, timeSince, escape

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--model_file','-m',type=str,default='../../models/HW3/model.pkl',help='Model save target.')
parser.add_argument('--n_epochs','-e',type=int,default=5,help='set the number of training epochs.')
parser.add_argument('--learning_rate','-lr',type=float,default=0.01,help='set learning rate.')
parser.add_argument('--attn_type','-at',type=str,default='hard',help='attention type')
args = parser.parse_args()
# You can add MIN_FREQ, MAX_LEN, and BATCH_SIZE as args too

n_epochs = args.n_epochs
learning_rate = args.learning_rate
attn_type = args.attn_type

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
DE = data.Field(tokenize=tokenize_de)
EN = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS

MAX_LEN = 20
train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                         len(vars(x)['trg']) <= MAX_LEN)

print(train.fields)
print(len(train))
print(vars(train[0]))

MIN_FREQ = 5
DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)
print(DE.vocab.freqs.most_common(10))
print("Size of German vocab", len(DE.vocab))
print(EN.vocab.freqs.most_common(10))
print("Size of English vocab", len(EN.vocab))
print(EN.vocab.stoi["<s>"], EN.vocab.stoi["</s>"]) # vocab index for <s>, </s>

BATCH_SIZE = 32
train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                  repeat=False, sort_key=lambda x: len(x.src))

batch = next(iter(train_iter))
print("Source")
print(batch.src)
print("Target")
print(batch.trg)

url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
EN.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) # feel free to alter path
print("Word embeddings size ", EN.vocab.vectors.size())
word2vec = EN.vocab.vectors
print("REMINDER!!! Did you create ../../models/HW3?????")

sos_token = EN.vocab.stoi["<s>"]
eos_token = EN.vocab.stoi["</s>"]

''' TODO
Study the data form. In training I assume batch.trg has last column of all </s>. Is this true?
- If not, how am I gonna handle training on uneven batches, where sentences finish at different lengths?
Predict function hack ideas? Involving MAX_LEN or eos_token


How is ppl calculated with no teacher forcing? I don't think you can. Just abstain from teacher forcing for now
- Build a section-code-style s2s without teacher forcing
- If we have time, we can try the tutorial script with and without attn, see if teacher forcing makes a difference
Yes, there is a German word2vec and I should use it
BLEU perl script
What is purpose of baseline reward?
BSO is for all the models, cause you search through a graph of words. (although searching for z's has been studied)
Can I throw out the perplexity from predicting on <s>? Who knows what the first word in a sentence is?

Pass a binary mask to attention module...? 

Consult papers for hyperparameters
Multi-layer, bidirectional, LSTM instead of GRU, etc
Dropout, embedding max norms, etc
'''


model = AttnNetwork()
model.cuda()

start = time.time()
print_every = 100
plot_every = 100
plot_losses = []
avg_acc = 0
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    train_iter.init_epoch()
    ctr = 0
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    for batch in iter(train_iter):
        ctr += 1
        optim.zero_grad()
        x_de = batch.src.cuda()
        x_en = batch.trg.cuda()
        loss, neg_reward = model.forward(x_de, x_en, attn_type)
        y_pred = model.predict(x_de, attn_type)
        # lesser_of_two_evils = min(y_pred.size(1),x_en.size(1)) # TODO: temporary fix!!
        # correct = torch.sum(y_pred[:,1:lesser_of_two_evils]==x_en[:,1:lesser_of_two_evils]) # exclude <s> token in acc calculation
        avg_acc = 0.95*avg_acc + 0.05*correct/(x_en.size(0)*x_en.size(1))
        (loss + neg_reward).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1) # TODO: is this right? it didn't work last time
        optim.step()
        print_loss_total += loss / x_en.size(1)
        plot_loss_total += loss / x_en.size(1)

        if ctr % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            timenow = timeSince(start)
            print ('Time %s, Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, Reward: %.2f, Accuracy: %.2f, PPL: %.2f' 
                %(timenow, epoch+1, num_epochs, ctr, len(train_iter), print_loss_avg,
                    model.baseline.data[0], avg_acc, np.exp(print_loss_avg)))

        if ctr % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    val_loss_total = 0 # Validation/early stopping
    for batch in iter(val_iter):
        x_de = batch.src.cuda()
        x_en = batch.trg.cuda()
        loss, neg_reward = model.forward(x_de, x_en, attn_type, update_baseline=False)
        # too lazy to implement reward or accuracy for validation
        val_loss_total += loss / x_en.size(1)
    val_loss_avg = val_loss_total / len(val_iter)
    timenow = timeSince(start)
    print('Validation. Time %s, PPL: %.2f' %(timenow, np.exp(val_loss_avg)))

# NOTE: AttnNetwork averages loss within batches, but neither over sentences nor across batches. thus, rescaling is necessary

with open("preds.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['id','word'])
    for i, l in enumerate(open("source_test.txt"),1):
        ls = [[DE.vocab.stoi[w] for w in l.split(' ')]] # no eos or sos. size 1,n_de
        x_de = Variable(torch.cuda.LongTensor(ls))
        _,wordlist,_ = model.predict2(x_de,beamsz=100,gen_len=3)
        # wordlist is beamsz,3
        longstr = ' '.join(['|'.join([EN.vocab.itos[w] for w in beam]) for beam in wordlist])
        longstr = escape(longstr)
        writer.writerow([i,longstr])

torch.save(model.state_dict(), args.model_file)
# showPlot(plot_losses) # TODO: function not added/checked

# visualize only for AttnNetwork
def visualize(attns,sentence_de,bs,nwords,flname): # attns = (SentLen_EN)x(SentLen_DE), sentence_de = ["German_1",...,"German_(SentLen_DE)"]
    _,wordlist,attns = model.predict2(sentence_de,beamsz=bs,gen_len=nwords)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attns.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(sentence_de, rotation=90)
    ax.set_yticklabels(wordlist)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("{}.png".format(flname))

list_of_german_sentences = [[""]]

cntr = 0
for sentence_de in list_of_german_sentences:
    flname = "plot_"+"{}".format(cntr)
    visualize(model,sentence_de,5,10,"{}".format(flname))
    cntr += 1

