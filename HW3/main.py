import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions # this package provides a lot of nice abstractions for policy gradients
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors
import spacy
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

import csv
import time
import argparse

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--model_type','-m',type=int,default=0,help='Model type (0 for Attn, 1 for S2S)')
parser.add_argument('--model_file','-mf',type=str,default='../../models/HW3/model.pkl',help='Model save target.')
parser.add_argument('--n_epochs','-e',type=int,default=3,help='set the number of training epochs.')
parser.add_argument('--learning_rate','-lr',type=float,default=0.01,help='set learning rate.')
parser.add_argument('--attn_type','-at',type=str,default='soft',help='attention type')
parser.add_argument('--clip_constraint','-cc',type=float,default=5.0,help='weight norm clip constraint')
parser.add_argument('--word2vec','-w',action='store_true',help='Raise flag to initialize with word2vec embeddings')
parser.add_argument('--embedding_dims','-ed',type=int,default=300,help='dims for word2vec embeddings')
parser.add_argument('--hidden_depth','-hd',type=int,default=1,help='Number of hidden layers in encoder/decoder')
parser.add_argument('--hidden_size','-hs',type=int,default=500,help='Size of each hidden layer in encoder/decoder')
parser.add_argument('--vocab_layer_dim','-vd',type=int,default=500,help='Size of hidden vocab layer transformation')
parser.add_argument('--weight_tying','-wt',action='store_true',help='Raise flag to engage weight tying')
parser.add_argument('--bidirectional','-b',action='store_true',help='Raise to make encoder bidirectional')
args = parser.parse_args()
# You can add MIN_FREQ, MAX_LEN, and BATCH_SIZE as args too

model_type = args.model_type
n_epochs = args.n_epochs
learning_rate = args.learning_rate
attn_type = args.attn_type
clip_constraint = args.clip_constraint
word2vec = args.word2vec

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

print("Getting datasets!")

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
print("Source size", batch.src.size())
print("Target size", batch.trg.size())

# https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
if word2vec:
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    EN.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) # feel free to alter path
    print("Simple English embeddings size", EN.vocab.vectors.size())
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec'
    DE.vocab.load_vectors(vectors=Vectors('wiki.de.vec', url=url)) # feel free to alter path
    print("German embeddings size", DE.vocab.vectors.size())

print("REMINDER!!! Did you create ../../models/HW3?????")

sos_token = EN.vocab.stoi["<s>"]
eos_token = EN.vocab.stoi["</s>"]
pad_token = EN.vocab.stoi["<pad>"]

''' TODO
ELBY
volatile=True
Padding?
- Test NLLLoss with ignore_index and size_average=False
- Simplify loss (target) masking with cross entropy
- Use a binary mask to zero out attention to paddings in the source.
Evaluate S2S, create predict2 for S2S.
Plot attention

EXTENSIONS
Consult papers for hyperparameters
Multi-layer, bidirectional (see Piazza), GRU instead of LSTM
Pretrained embeddings
Weight tying, interpolation
Dropout, embedding max norms, weight clipping, learning rate scheduling, residual connections
Hard attention, with updating baseline
More complex regularization techniques (Yoon piazza)
Checkout openNMT for inspiration

ANCILLARY
If we have time, we can try the pytorch tutorial script with and without attn, to see if teacher forcing makes a difference
Predict function hack ideas? Involving MAX_LEN or eos_token
BLEU perl script
Can I throw out the perplexity from predicting on <s>? Who knows what the first word in a sentence is?
How to run jupyter notebooks in cloud?
Generate longer full sentences with small beams. Not fixed-length.

QUESTIONS
Yoon: why do you average loss over minibatches but not over time?
Justin's piazza post: speedup by batching over time?
What's purpose of baseline? Your code is wrong- subtract something averaged over bs & n_de from something averaged over bs?
'''
from models import AttnNetwork, CandList, S2S
from helpers import asMinutes, timeSince, escape, flip

if model_type == 0:
    model = AttnNetwork(word_dim=args.embedding_dims, n_layers=args.hidden_depth, hidden_dim=args.hidden_size,
                        vocab_layer_dim=args.vocab_layer_dim, weight_tying=args.weight_tying)
elif model_type == 1:
    model = S2S()
model.cuda()

start = time.time()
print_every = 100
plot_every = 100
plot_losses = []
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    train_iter.init_epoch()
    ctr = 0
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_acc_total = 0 # Reset every print_every
    for batch in iter(train_iter):
        ctr += 1
        optimizer.zero_grad()
        x_de = batch.src.transpose(1,0).cuda() # bs,n_de
        x_en = batch.trg.transpose(1,0).cuda() # bs,n_en
        if model_type == 1:
            x_de = flip(x_de,1) # reverse direction
        loss, reinforce_loss = model.forward(x_de, x_en, attn_type)
        print_loss_total += loss.data[0] / x_en.size(1)
        plot_loss_total += loss.data[0] / x_en.size(1)
        # TODO: this is underestimating PPL! It divides by x_en when it should divide by no_pad

        # JUST COMMENTING THIS OUT TO TEST S2S! TODO: uncomment this once s2s predict methods are surgically replaced
        # y_pred,_ = model.predict(x_de, x_en, attn_type) # bs,n_en
        # correct = (y_pred == x_en)
        # no_pad = (x_en != pad_token) & (x_en != sos_token)
        # print_acc_total += (correct & no_pad).data.sum() / no_pad.data.sum()

        (loss + reinforce_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip_constraint) # TODO: is this right? it didn't work last time
        optimizer.step()
        if args.weight_tying:
            model.vocab_layer.e2v.weight.data.copy_(model.embedding_en.weight.data)
        
        if ctr % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_acc_avg = print_acc_total / print_every
            print_acc_total = 0
            timenow = timeSince(start)
            print ('Time %s, Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, Reward: %.2f, Accuracy: %.2f, PPL: %.2f' 
                %(timenow, epoch+1, n_epochs, ctr, len(train_iter), print_loss_avg,
                    model.baseline.data[0], print_acc_avg, np.exp(print_loss_avg)))

        if ctr % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_total = 0
            plot_losses.append(plot_loss_avg)

    torch.save(model.state_dict(), args.model_file) # I'm Paranoid!!!!!!!!!!!!!!!!
    val_loss_total = 0 # Validation/early stopping
    for batch in iter(val_iter):
        x_de = batch.src.transpose(1,0).cuda()
        x_en = batch.trg.transpose(1,0).cuda()
        loss, reinforce_loss = model.forward(x_de, x_en, attn_type)
        # too lazy to implement reward or accuracy for validation
        val_loss_total += loss.data[0] / x_en.size(1)
    val_loss_avg = val_loss_total / len(val_iter)
    timenow = timeSince(start)
    print('Validation. Time %s, PPL: %.2f' %(timenow, np.exp(val_loss_avg)))

# NOTE: AttnNetwork averages loss within batches, but neither over sentences nor across batches. thus, rescaling is necessary

torch.save(model.state_dict(), args.model_file)

with open("preds.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['id','word'])
    for i, l in enumerate(open("source_test.txt"),1):
        ls = [[DE.vocab.stoi[w] for w in l.split(' ')]] # no eos or sos. size 1,n_de
        x_de = Variable(torch.cuda.LongTensor(ls))
        _,wordlist,_ = model.predict2(x_de,beamsz=100,gen_len=3)
        # wordlist is beamsz,3
        longstr = ' '.join(['|'.join([EN.vocab.itos[w] for w in trigram]) for trigram in wordlist])
        longstr = escape(longstr)
        writer.writerow([i,longstr])

# showPlot(plot_losses) # TODO: function not added/checked
# # visualize only for AttnNetwork
# def visualize(attns,sentence_de,bs,nwords,flname): # attns = (SentLen_EN)x(SentLen_DE), sentence_de = ["German_1",...,"German_(SentLen_DE)"]
#     _,wordlist,attns = model.predict2(sentence_de,beamsz=bs,gen_len=nwords)

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attns.numpy(), cmap='bone')
#     fig.colorbar(cax)

#     # Set up axes
#     ax.set_xticklabels(sentence_de, rotation=90)
#     ax.set_yticklabels(wordlist)

#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     plt.savefig("{}.png".format(flname))

# list_of_german_sentences = [[""]]

# cntr = 0
# for sentence_de in list_of_german_sentences:
#     flname = "plot_"+"{}".format(cntr)
#     visualize(model,sentence_de,5,10,"{}".format(flname))
#     cntr += 1

