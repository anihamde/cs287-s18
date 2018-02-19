import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import spacy

import csv
import time
import copy
import argparse

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
print(EN.vocab.stoi["<s>"], EN.vocab.stoi["</s>"]) #vocab index for <s>, </s>

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
print("REMINDER!!! Did you create ../../models/HW2?????")

''' TODO
Pass a binary mask to attention module
How is ppl calculated? How does bleu perl script work?
can I contain encoder/decoder in same network? and not use a for loop?
'''


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, bs=BATCH_SIZE):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bs = bs
        self.embedding = nn.Embedding(word2vec.size(0), word2vec.size(1))
        self.embedding.weight.data.copy_(word2vec)
        self.gru = nn.GRU(word2vec.size(1), hidden_size) # input sz,hidden sz,nlayers

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs).view(1,self.bs,-1) # 1,bs,300
        output = embedded
        output, hidden = self.gru(output, hidden) # output 1,bs,hiddensz*ndirections
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, self.bs, self.hidden_size)) # nlayers*ndirections,bs,hiddensz
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, bs=BATCH_SIZE): # dropout?
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bs = bs
        self.embedding = nn.Embedding(len(DE.vocab), 300) # 300??
        self.gru = nn.GRU(300, hidden_size)
        self.out = nn.Linear(hidden_size, 1) # output_size should be 1
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, inputs, hidden):
        output = self.embedding(inputs).view(1,self.bs,-1) # 1,bs,300
        output = F.relu(output)
        output, hidden = self.gru(output, hidden) # output 1,bs,hiddensz*directions
        output = self.out(output[0]) # bs,1
        output = self.softmax(output.squeeze(1)) # bs
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result





teacher_forcing_ratio = 0.5

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LEN):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size(0) # input is maxlen,bs
    target_length = target_variable.size(0) # target is maxlen-1,bs

    # check all this, check decoder too
    encoder_outputs = Variable(torch.zeros(max_length, BATCH_SIZE, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if torch.cuda.is_available() else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]

    sos_token = EN.vocab.stoi["<s>"]
    decoder_input = Variable(torch.LongTensor([sos_token]*BATCH_SIZE)) # should be bs-dimensional
    decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder( # attention!!
            #     decoder_input, decoder_hidden, encoder_outputs)
            decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
import time
import math
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    ctr = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in num_epochs:
        train_iter.init_epoch()
        for batch in iter(train_iter):
            ctr += 1
            # changed!
            input_variable = batch.src.transpose(1,0)
            target_variable = batch.trg.transpose(1,0)
            loss = train(input_variable, target_variable, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if ctr % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, ctr / n_iters),
                                             ctr, ctr / n_iters * 100, print_loss_avg))

            if ctr % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


hidden_size = 256 # idk?
# got rid of input_size arg
encoder = EncoderRNN(hidden_size)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)
if torch.cuda.is_available():
    encoder = encoder.cuda()
    decoder = decoder.cuda()

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)