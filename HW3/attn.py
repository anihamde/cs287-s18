import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions # this package provides a lot of nice abstractions for policy gradients
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
import spacy

import csv
import time
import math
import copy
import argparse

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--model_file','-m',type=str,default='../../models/HW3/model.pkl',help='Model save target.')
args = parser.parse_args()
# TODO: bs = args.batch_size etc

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

# url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
# EN.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) # feel free to alter path
# print("Word embeddings size ", EN.vocab.vectors.size())
# word2vec = EN.vocab.vectors
print("REMINDER!!! Did you create ../../models/HW3?????")

sos_token = EN.vocab.stoi["<s>"]
eos_token = EN.vocab.stoi["</s>"]

# TODO: consult paper for hyperparams
# TODO: beam search optim. does it require a whole new method?
# TODO: bleu calculation

class AttnNetwork(nn.Module):
    def __init__(self, word_dim=300, hidden_dim=500):
        super(AttnNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        # LSTM initialization params: inputsz,hiddensz,nlayers,bias,batch_first,bidirectional
        self.encoder = nn.LSTM(word_dim, hidden_dim, num_layers = 1, batch_first = True)
        self.decoder = nn.LSTM(word_dim, hidden_dim, num_layers = 1, batch_first = True)
        self.embedding_de = nn.Embedding(len(DE.vocab), word_dim)
        self.embedding_en = nn.Embedding(len(EN.vocab), word_dim)
        # vocab layer will combine dec hidden state with context vector, and then project out into vocab space 
        self.vocab_layer = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                         nn.Tanh(), nn.Linear(hidden_dim, vocab_size), nn.LogSoftmax())
        # baseline reward, which we initialize with log 1/V
        self.baseline = Variable(torch.zeros(1).fill_(np.log(1/vocab_size)))                
        
    def forward(self, x_de, x_en, attn_type="hard", bs=BATCH_SIZE, update_baseline=True):
        # x_de is bs,n_de. x_en is bs,n_en
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en) # bs,n_en,word_dim
        h0 = Variable(torch.zeros(1, bs, self.hidden_dim)) 
        c0 = Variable(torch.zeros(1, bs, self.hidden_dim))
        # hidden vars have dimension nlayers*ndirections,bs,hiddensz
        enc_h, _ = self.encoder(emb_de, (h0, c0))
        # enc_h is bs,n_de,hiddensz*ndirections. ordering is different from last week because batch_first=True
        dec_h, _ = self.decoder(emb_en, (h0, c0))
        # dec_h is bs,n_en,hidden_size*ndirections
        # we've gotten our encoder/decoder hidden states so we are ready to do attention        
        # first let's get all our scores, which we can do easily since we are using dot-prod attention
        scores = torch.bmm(enc_h, dec_h.transpose(1,2)) 
        # (bs,n_de,hiddensz*ndirections) * (bs,hiddensz*ndirections,n_en) = (bs,n_de,n_en)
        neg_reward = 0 # we only use this variable for hard attention
        loss = 0
        avg_reward = 0
        # we just iterate to dec_h.size(1)-1, since there's </s> at the end of each sentence
        for t in range(dec_h.size(1)-1): # iterate over english words, with teacher forcing
            attn_dist = F.softmax(scores[:, :, t], dim=1) # bs,n_de. these are the alphas (attention scores for each german word)
            if attn_type == "hard":
                cat = torch.distributions.Categorical(attn_dist) 
                attn_samples = cat.sample() # bs. each element is a sample from categorical distribution
                one_hot = Variable(torch.zeros_like(attn_dist.data).scatter_(-1, attn_samples.data.unsqueeze(1), 1)) # bs,n_de
                # made a bunch of one-hot vectors
                context = torch.bmm(one_hot.unsqueeze(1), enc_h).squeeze(1)
                # now we use the one-hot vectors to select correct hidden vectors from enc_h
                # (bs,1,n_de) * (bs,n_de,hiddensz*ndirections) = (bs,1,hiddensz*ndirections). squeeze to bs,hiddensz*ndirections
            else:
                context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1) # same dimensions
            # context is bs,hidden_size*ndirections
            # the rnn output and the context together make the decoder "hidden state", which is bs,2*hidden_size*ndirections
            pred = self.vocab_layer(torch.cat([dec_h[:, t], context], 1)) # bs,len(EN.vocab)
            y = x_en[:, t+1] # bs. these are our labels
            reward = torch.gather(pred, 1, y.unsqueeze(1)) # bs
            # reward[i] = pred[i,y[i]]. this gets log prob of correct word for each batch. similar to -crossentropy
            avg_reward += reward.data.mean()
            if attn_type == "hard":
                neg_reward -= (cat.log_prob(attn_samples) * (reward.detach()-self.baseline)).mean() 
                # reinforce rule (just read the formula), with special baseline
            loss -= reward.mean() # minimizing loss is maximizing reward
        avg_reward = avg_reward/dec_h.size(1)
        if update_baseline: # update baseline as a moving average
            self.baseline.data = 0.95*self.baseline.data + 0.05*avg_reward
        return loss, neg_reward
    
    def predict(self, x_de, attn_type = "hard", bs=BATCH_SIZE):
        # predict with greedy decoding
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        h = Variable(torch.zeros(1, bs, self.hidden_dim))
        c = Variable(torch.zeros(1, bs, self.hidden_dim))
        enc_h, _ = self.encoder(emb_de, (h, c))
        # all the same. enc_h is bs,n_de,hiddensz*ndirections. h and c are both nlayers*ndirections,bs,hiddensz
        y = [Variable(torch.LongTensor([sos.token]*bs))] # bs
        self.attn = []
        n_en = MAX_LEN # this will change
        for t in range(n_en): # generate some english.
            emb_t = self.embedding(y[-1]) # embed the last thing we generated. bs
            dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c)) # dec_h is bs,1,hiddensz*ndirections (batch_first=True)
            scores = torch.bmm(enc_h, dec_h.transpose(1,2)).squeeze(2)
            # (bs,n_de,hiddensz*ndirections) * (bs,hiddensz*ndirections,1) = (bs,n_de,1). squeeze to bs,n_de
            attn_dist = F.softmax(scores,dim=1)
            self.attn.append(attn_dist.data)
            if attn_type == "hard":
                _, argmax = attn_dist.max(1) # bs. for each batch, select most likely german word to pay attention to
                one_hot = Variable(torch.zeros_like(attn_dist.data).scatter_(-1, argmax.data.unsqueeze(1), 1))
                context = torch.bmm(one_hot.unsqueeze(1), enc_h).squeeze(1)
            else:
                context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1)
            # the difference btwn hard and soft is just whether we use a one_hot or a distribution
            # context is bs,hiddensz*ndirections
            pred = self.vocab_layer(torch.cat([dec_h.squeeze(1), context], 1)) # bs,len(EN.vocab)
            _, next_token = pred.max(1) # bs
            y.append(next_token)
        self.attn = torch.stack(self.attn, 0).transpose(0, 1) # bs,n_en,n_de (for visualization!)
        return torch.stack(y, 0).transpose(0, 1) # bs,n_en




def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since):
    now = time.time()
    s = now - since
    # es = s / (percent)
    # rs = es - s
    return '%s' % (asMinutes(s))

model = AttnNetwork()
n_epochs = 10
print_every = 100
plot_every = 100
learning_rate = 0.01
attn_type = "hard"

start = time.time()
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
        x_de = batch.src
        x_en = batch.trg
        loss, neg_reward = model.forward(x_de, x_en, attn_type)
        y_pred = model.predict(x_de, attn_type)
        lesser_of_two_evils = min(y_pred.size(1),x_en.size(1)) # TODO: temporary fix!!
        correct = torch.sum(y_pred[:,1:lesser_of_two_evils]==x_en[:,1:lesser_of_two_evils]) # exclude <s> token in acc calculation
        avg_acc = 0.95*avg_acc + 0.05*correct/(x_en.size(0)*x_en.size(1))
        (loss + neg_reward).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1)
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
        x_de = batch.src
        x_en = batch.trg
        loss, neg_reward = model.forward(x_de, x_en, attn_type, update_baseline=False)
        # too lazy to implement reward or accuracy for validation
        val_loss_total += loss / x_en.size(1)
    val_loss_avg = val_loss_total / len(val_iter)
    timenow = timeSince(start)
    print('Validation. Time %s, PPL: %.2f' %(timenow, np.exp(val_loss_avg)))

# NOTE: AttnNetwork averages loss within batches, but neither over sentences nor across batches. thus, rescaling is necessary

torch.save(model.state_dict(), args.model_file)
showPlot(plot_losses)
