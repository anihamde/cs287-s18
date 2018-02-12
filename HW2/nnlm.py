import numpy as np
from collections import Counter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv
import argparse

parser = argparse.ArgumentParser(description='nnlm training runner')
parser.add_argument('--model_file','-m',type=str,default='../../models/HW2/nnlm.pkl',help='Model save target.')
parser.add_argument('--batch_size','-bs',type=int,default=10,help='set training batch size. default = 10.')
parser.add_argument('--receptive_field','-rf',type=int,default=5,help='set receptive field of nnlm.')
parser.add_argument('--hidden_size','-hs',type=int,default=100,help='set size of hidden layer.')
parser.add_argument('--learning_rate','-lr',type=float,default=0.001,help='set learning rate.')
parser.add_argument('--weight_decay','-wd',type=float,default=0.0,help='set L2 normalization factor.')
parser.add_argument('--num_epochs','-e',type=int,default=10,help='set the number of training epochs.')
parser.add_argument('--embedding_max_norm','-emn',type=float,default=15,help='set max L2 norm of word embedding vector.')
parser.add_argument('--skip_training','-sk',action='store_true',help='raise flag to skip training and go to eval.')
args = parser.parse_args()

# Hyperparameters
bs = args.batch_size # batch size
n = args.receptive_field # receptive field
hidden_size = args.hidden_size
learning_rate = args.learning_rate
weight_decay = args.weight_decay
num_epochs = args.num_epochs
emb_mn = args.embedding_max_norm # embedding max norm (folk knowledge)

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

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) # feel free to alter path
print("Word embeddings size ", TEXT.vocab.vectors.size())
print("REMINDER!!! Did you create ../../models/HW2?????")

# TODO: Is 8000 ppl reasonable? Why are my perplexities so high? What separates us and Bengio???
# compare to lstm/trigram ppls? print ppls of individual batches? or don't validate on full bptt_len?
# also, do we have the right idea for precision?
# the lstm losses are also screwed. are the lstm's sentences at least reasonable?
# TODO: hidden to 300, then to |V|
# TODO: mixture of models with interpolated trigram (fixed or learned weights)
# TODO: bengio's idea, set w to zero

# TODO: reg: want to try weight clippings too? dropout maybe?
# TODO: energy min network, what else?
# TODO: our own extensions (multichannel with glove, static/dynamic, etc?) (conv layers) (dropout) (recurrence) (pad at the beginning?)

class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        # Test the max_norm. Is it norm per row, or total norm of the whole matrix?
        self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1),max_norm=emb_mn)
        self.embeddings.weight.data = TEXT.vocab.vectors
        self.h = nn.Linear(n*300,hidden_size)
        self.u = nn.Linear(hidden_size,len(TEXT.vocab))
        self.w = nn.Linear(n*300,len(TEXT.vocab))
    def forward(self, inputs): # inputs (batch size, "sentence" length) bs,n
        embeds = self.embeddings(inputs) # bs,n,300
        embeds = embeds.view(-1,n*300) # bs,n*300
        out = F.tanh(self.h(embeds)) # bs,hidden_size
        out = self.u(out) # bs,|V|
        out += self.w(embeds) # bs,|V|
        #out = F.softmax(out,dim=1)
        return out

model = NNLM()
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
biasparams = []
weightparams = []
for name,param in model.named_parameters():
    if 'bias' in name:
        biasparams.append(param)
    else:
        weightparams.append(param)
optimizer = torch.optim.Adam([
                {'params': biasparams},
                {'params': weightparams, 'weight_decay':weight_decay}
            ], lr=learning_rate)

# This whole thing takes about half a minute on GPU
def validate():
    model.eval()
    correct = total = 0
    precisionmat = (1/np.arange(1,21))[::-1].cumsum()[::-1]
    precisionmat = torch.cuda.FloatTensor(precisionmat.copy()) # hm
    precision = 0
    crossentropy = 0
    for batch in iter(val_iter):
    sentences = batch.text.transpose(1,0).cuda() # bs, n
        if sentences.size(1) < n+1: # make sure sentence length is long enough
            pads = Variable(torch.zeros(sentences.size(0),n+1-sentences.size(1))).type(torch.cuda.LongTensor)
            sentences = torch.cat([pads,sentences],dim=1)
        for j in range(n,sentences.size(1)):
            out = model(sentences[:,j-n:j]) # bs,|V|
            labels = sentences[:,j] # bs
            # cross entropy
            crossentropy += F.cross_entropy(out,labels)
            # precision
            out, labels = out.data, labels.data
            _, outsort = torch.sort(out,dim=1,descending=True)
            outsort = outsort[:,:20]
            inds = (outsort-labels.unsqueeze(1)==0)
            inds = inds.sum(dim=0).type(torch.cuda.FloatTensor)
            precision += inds.dot(precisionmat)
            # plain ol accuracy
            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum()
            # if total % 500 == 0:
                # DEBUGGING: see the rest in trigram.py
                # print('we are on example', total)
                # for s in range(bs):
                #     print([TEXT.vocab.itos[w] for w in sentences[s,j-n:j].data])
                #     print(TEXT.vocab.itos[labels[s]])
                #     print([TEXT.vocab.itos[w] for w in outsort[s]])
                # print('Test Accuracy', correct/total)
                # print('Precision',precision/total)
                # print('Perplexity',torch.exp(bs*crossentropy/total).data[0])
    return correct/total, precision/total, torch.exp(bs*crossentropy/total).data[0]
    # test acc, precision, ppl
    # F.cross_entropy averages instead of adding


if not args.skip_training:
    losses = []
    for i in range(num_epochs):
        model.train()
        ctr = 0
        for batch in iter(train_iter):
            # print('TEST DELETE THIS embedding norm', model.embeddings.weight.norm())
            sentences = batch.text.transpose(1,0).cuda() # bs,n
            if sentences.size(1) < n+1: # make sure sentence length is long enough
                pads = Variable(torch.zeros(sentences.size(0),n+1-sentences.size(1))).type(torch.cuda.LongTensor)
                sentences = torch.cat([pads,sentences],dim=1)
            for j in range(n,sentences.size(1)):
                out = model(sentences[:,j-n:j])
                loss = criterion(out,sentences[:,j])
                model.zero_grad()
                loss.backward()
                optimizer.step()
            ctr += 1
            if ctr % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                    %(i+1, num_epochs, ctr, len(train_iter), loss.data[0]))
            losses.append(loss.data[0])

        # can add a net_flag to these file names. and feel free to change the paths
        np.save("../../models/HW2/nnlm_losses.npy",np.array(losses))
        torch.save(model.state_dict(), args.model_file)
        # for early stopping
        acc, prec, ppl = validate()
        print("Val acc, prec, ppl", acc, prec, ppl)
else:
    model.load_state_dict(torch.load(args.model_file))


model.eval()
with open("nnlm_predictions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['id','word'])
    for i, l in enumerate(open("input.txt"),1):
        words = [TEXT.vocab.stoi[word] for word in l.split(' ')]
        words = Variable(torch.cuda.LongTensor(words[-1-n:-1]))
        out = model(words)
        _, predicted = torch.sort(out,dim=1,descending=True)
        predicted = predicted[0,:20].data.tolist()
        predwords = [TEXT.vocab.itos[x] for x in predicted]
        writer.writerow([i,' '.join(predwords)])
