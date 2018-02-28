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
import yaml
from collections import OrderedDict
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

import csv
import time
import argparse

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--model_type','-m',type=int,default=0,help='Model type (0 for AttnLSTM, 1 for S2S, 2 for AttnGRU)')
parser.add_argument('--model_file','-mf',type=str,default='../../models/HW3/model.pkl',help='Model save target.')
parser.add_argument('--architecture_file','-y',type=str,default='../../models/HW3/model.yaml',help='YAML file containing specs to build model.')
parser.add_argument('--n_epochs','-e',type=int,default=3,help='set the number of training epochs.')
parser.add_argument('--adadelta','-ada',action='store_true',help='Use Adadelta optimizer')
parser.add_argument('--learning_rate','-lr',type=float,default=0.01,help='set learning rate.')
parser.add_argument('--rho','-r',type=float,default=0.95,help='rho for Adadelta optimizer')
parser.add_argument('--weight_decay','-wd',type=float,default=0.0,help='Weight decay constant for optimizer')
parser.add_argument('--accuracy','-acc',action='store_true',help='Calculate accuracy during training loop.')
parser.add_argument('--frequent_ckpt','-ckpt',action='store_true',help='Save checkpoints every epoch, instead of just at the end.')
parser.add_argument('--attn_type','-at',type=str,default='soft',help='attention type')
parser.add_argument('--clip_constraint','-cc',type=float,default=5.0,help='weight norm clip constraint')
parser.add_argument('--word2vec','-w',action='store_true',help='Raise flag to initialize with word2vec embeddings')
parser.add_argument('--embedding_dims','-ed',type=int,default=300,help='dims for word2vec embeddings')
parser.add_argument('--hidden_depth','-hd',type=int,default=1,help='Number of hidden layers in encoder/decoder')
parser.add_argument('--hidden_size','-hs',type=int,default=500,help='Size of each hidden layer in encoder/decoder')
parser.add_argument('--vocab_layer_size','-vs',type=int,default=500,help='Size of hidden vocab layer transformation')
parser.add_argument('--weight_tying','-wt',action='store_true',help='Raise flag to engage weight tying')
parser.add_argument('--bidirectional','-b',action='store_true',help='Raise to make encoder bidirectional')
parser.add_argument('--LSTM_dropout','-ld',type=float,default=0.0,help='Dropout rate inside encoder/decoder LSTMs')
parser.add_argument('--vocab_layer_dropout','-vd',type=float,default=0.0,help='Dropout rate in vocab layer')
parser.add_argument('--interpolated_model','-i',action='store_true',help="Invoke interpolated model, above architecture defining args suppressed, below args activated.")
parser.add_argument('--saved_parameters','-sp',type=str,nargs='+',help="List of model parameter files (PKLs). Needs to match '--saved_architectures' arg.")
parser.add_argument('--saved_architectures','-sa',type=str,nargs='+',help="List of model architecture files (YAMLs). Needs to match '--saved_parameters' arg.")
parser.add_argument('--convolution_embedding_size','-ces',type=int,default=300,help='size of embedding in Alpha.')
parser.add_argument('--convolutional_featuremap_1','-cf1',type=int,default=200,help='Featuremap density for 3x1 conv.')
parser.add_argument('--convolutional_featuremap_2','-cf2',type=int,default=200,help='Featuremap density for 5x1 conv.')
parser.add_argument('--alpha_dropout','-ad',type=float,default=0.5,help='Dropout for Alpha.')
parser.add_argument('--alpha_linear_size','-cls',type=int,default=200,help='Size of hidden fully connected layer in Alpha')
parser.add_argument('--freeze_models','-fz',action='store_true',help='raise flag to freeze ensemble member parameters.')
args = parser.parse_args()
# You can add MIN_FREQ, MAX_LEN, and BATCH_SIZE as args too

model_type = args.model_type
n_epochs = args.n_epochs
learning_rate = args.learning_rate
attn_type = args.attn_type
clip_constraint = args.clip_constraint
word2vec = args.word2vec
rho = args.rho
weight_decay = args.weight_decay

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

unk_token = EN.vocab.stoi["<unk>"]
pad_token = EN.vocab.stoi["<pad>"]
sos_token = EN.vocab.stoi["<s>"]
eos_token = EN.vocab.stoi["</s>"]

''' TODO
Fix bidirectional S2S
Does ppl change if you average loss the actual Yoon way?
Try plotting train and val acc after each batch.
BLEU perl script

LATEX
S2S results, with 1 and 4 layers. And bidirectional S2S
Sweep over hidden sizes and num layers.
Bidirectional vs not bidirectional (easily helps).
Baseline, word2vec, weight tying+word2vec.
GRU vs LSTM
Hard attention, with updating baseline
Interpolation

ANCILLARY
Dropout, embedding max norms, weight clipping, learning rate scheduling (ada), residual connections
More complex regularization techniques (Yoon piazza)
Checkout openNMT for inspiration
Bigger validation set
If we have time, we can try the pytorch tutorial script with and without attn, to see if teacher forcing makes a difference
How to run jupyter notebooks in cloud?
Generate longer full sentences with small beams. Not fixed-length.
''' 
from models import AttnNetwork, CandList, S2S, AttnGRU
from helpers import asMinutes, timeSince, escape, flip, freeze_model

if args.interpolated_model:
    architecture_dict = OrderedDict([ ('model_type','interpolated'),
        ('embedding_features',args.convolution_embedding_size),('n_featmaps1',args.convolutional_featuremap_1),
        ('n_featmaps2',args.convolutional_featuremap_2),('linear_size',args.alpha_linear_size),
        ('dropout_rate',args.alpha_dropout),('word2vec',args.word2vec),
        ('freeze_models',args.freeze_models)
    ])
    #
    with open(args.architecture_file,'w') as fh:
        [ fh.write("{}: {}\n".format(key,value)) for key,value in architecture_dict.items() ]
        fh.close()
    model_list = []
    arch_tuple = ( yaml.safe_load(open(x,'r')) for x in args.saved_architectures )
    for arch,params in zip(arch_tuple,args.saved_parameters):
        if arch['model_type'] == 0:
            model_grab = AttnNetwork(word_dim=arch['word_dim'], n_layers=arch['n_layers'], 
                                          hidden_dim=arch['hidden_dim'], word2vec=arch['word2vec'],
                                          vocab_layer_size=arch['vocab_layer_size'], LSTM_dropout=arch['LSTM_dropout'], 
                                          vocab_layer_dropout=arch['vocab_layer_dropout'], weight_tying=arch['weight_tying'], 
                                          bidirectional=arch['bidirectional'], attn_type=arch['attn_type'])
        elif arch['model_type'] == 1:
            model_grab = S2S(word_dim=arch['word_dim'], n_layers=arch['n_layers'], 
                                          hidden_dim=arch['hidden_dim'], word2vec=arch['word2vec'],
                                          vocab_layer_size=arch['vocab_layer_size'], LSTM_dropout=arch['LSTM_dropout'], 
                                          vocab_layer_dropout=arch['vocab_layer_dropout'], weight_tying=arch['weight_tying'], 
                                          bidirectional=arch['bidirectional'], attn_type=arch['attn_type'])
        elif arch['model_type'] == 3:
            model_grab = AttnGRU(word_dim=arch['word_dim'], n_layers=arch['n_layers'], 
                                          hidden_dim=arch['hidden_dim'], word2vec=arch['word2vec'],
                                          vocab_layer_size=arch['vocab_layer_size'], LSTM_dropout=arch['LSTM_dropout'], 
                                          vocab_layer_dropout=arch['vocab_layer_dropout'], weight_tying=arch['weight_tying'], 
                                          bidirectional=arch['bidirectional'], attn_type=arch['attn_type'])
        model_grab.load_state_dict(params)
        model_grab.cuda()
        model_list.append(model_grab)
    model_list = tuple(model_list)    
    model = Alpha(model_list, embedding_features=args.convolution_embedding_size, n_featmaps1=args.convolutional_featuremap_1, 
                  n_featmaps2=args.convolutional_featuremap_2, linear_size=args.alpha_linear_size, 
                  dropout_rate=args.alpha_dropout, word2vec=args.word2vec, 
                  freeze_models=args.freeze_models)
else:        
    architecture_dict = OrderedDict([ ('model_ype',args.model_type),
        ('word_dim',args.embedding_dims),('n_layers',args.hidden_depth),
        ('hidden_dim',args.hidden_size),('word2vec',args.word2vec),
        ('vocab_layer_size',args.vocab_layer_size),('LSTM_dropout',args.LSTM_dropout),
        ('vocab_layer_dropout',args.vocab_layer_dropout),('weight_tying',args.weight_tying),
        ('bidirectional',args.bidirectional),('attn_type',args.attn_type)
    ])
    #
    with open(args.architecture_file,'w') as fh:
        [ fh.write("{}: {}\n".format(key,value)) for key,value in architecture_dict.items() ]
        fh.close()
    #
    if model_type == 0:
        model = AttnNetwork(word_dim=args.embedding_dims, n_layers=args.hidden_depth, hidden_dim=args.hidden_size, word2vec=args.word2vec,
                            vocab_layer_size=args.vocab_layer_size, LSTM_dropout=args.LSTM_dropout, vocab_layer_dropout=args.vocab_layer_dropout, 
                            weight_tying=args.weight_tying, bidirectional=args.bidirectional, attn_type=attn_type)
    elif model_type == 1:
        model = S2S(word_dim=args.embedding_dims, n_layers=args.hidden_depth, hidden_dim=args.hidden_size, word2vec=args.word2vec,
                    vocab_layer_size=args.vocab_layer_size, LSTM_dropout=args.LSTM_dropout, vocab_layer_dropout=args.vocab_layer_dropout, 
                    weight_tying=args.weight_tying, bidirectional=args.bidirectional)
    if model_type == 2:
        model = AttnGRU(word_dim=args.embedding_dims, n_layers=args.hidden_depth, hidden_dim=args.hidden_size, word2vec=args.word2vec,
                        vocab_layer_size=args.vocab_layer_size, LSTM_dropout=args.LSTM_dropout, vocab_layer_dropout=args.vocab_layer_dropout, 
                        weight_tying=args.weight_tying, bidirectional=args.bidirectional, attn_type=attn_type)

model.cuda()

start = time.time()
print_every = 100
plot_every = 100
plot_losses = []
params = list(filter(lambda x: x.requires_grad, model.parameters()))
if (not args.freeze_models) and args.interpolated_model:
    for member in model.members:
        params.extend(list(filter(lambda x: x.requires_grad, member.parameters())))
if args.adadelta:
    optimizer = optim.Adadelta(params, lr=learning_rate, rho=rho, weight_decay=weight_decay)
else:
    optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)

for epoch in range(n_epochs):
    train_iter.init_epoch()
    ctr = 0
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_acc_total = 0 # Reset every print_every
    for batch in iter(train_iter):
        ctr += 1
        model.train()
        if (not args.freeze_models) and args.interpolated_model:
            for member in model.members:
                member.train()
        optimizer.zero_grad()
        x_de = batch.src.transpose(1,0).cuda() # bs,n_de
        x_en = batch.trg.transpose(1,0).cuda() # bs,n_en
        if model_type == 1:
            x_de = flip(x_de,1) # reverse direction
        loss, reinforce_loss, avg_reward, _ = model.forward(x_de, x_en)
        print_loss_total -= avg_reward
        plot_loss_total -= avg_reward

        # TODO: what happens if i don't multiply by x_en.size(1)
        loss *= x_en.size(1) # scaling purposes, ask yoon!
        reinforce_loss *= x_en.size(1)
        (loss + reinforce_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip_constraint)
        optimizer.step()
        if args.weight_tying:
            #model.vocab_layer.e2v.weight.data.copy_(model.embedding_en.weight.data)
            model.embedding_en.weight.data.copy_(model.vocab_layer.e2v.weight.data)

        if args.accuracy:
            model.eval()
            if (not args.freeze_models) and args.interpolated_model:
                for member in model.members:
                    member.eval()
            x_de = Variable(x_de.data, volatile = True)
            y_pred,_ = model.predict(x_de, x_en) # bs,n_en
            correct = (y_pred == x_en) # these are the same shape and both contain a sos_token row
            no_pad = (x_en != pad_token) & (x_en != sos_token)
            print_acc_total += (correct & no_pad).data.sum() / no_pad.data.sum()
        
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

    val_loss_total = 0 # Validation/early stopping
    model.eval()
    if (not args.freeze_models) and args.interpolated_model:
        for member in model.members:
            member.eval()
    for batch in iter(val_iter):
        x_de = batch.src.transpose(1,0).cuda()
        x_en = batch.trg.transpose(1,0).cuda()
        if model_type == 1:
            x_de = flip(x_de,1) # reverse direction
        x_de.volatile = True # "inference mode" supposedly speeds up
        loss, reinforce_loss, avg_reward, _ = model.forward(x_de, x_en)
        # too lazy to implement reward or accuracy for validation
        val_loss_total -= avg_reward
    val_loss_avg = val_loss_total / len(val_iter)
    timenow = timeSince(start)
    print('Validation. Time %s, PPL: %.2f' %(timenow, np.exp(val_loss_avg)))
    if args.frequent_ckpt:
        torch.save(model.state_dict(), args.model_file) # I'm Paranoid!!!!!!!!!!!!!!!!

torch.save(model.state_dict(), args.model_file)

model.eval()
if (not args.freeze_models) and args.interpolated_model:
    for member in model.members:
        member.eval()
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
# def visualize(sentence_de,bs,nwords,flname): # attns = (SentLen_EN)x(SentLen_DE), sentence_de = ["German_1",...,"German_(SentLen_DE)"]
#     ls = [[DE.vocab.stoi[w] for w in sentence_de.split(' ')]]
#     _,wordlist,attns = model.predict2(Variable(torch.cuda.LongTensor(ls)),beamsz=bs,gen_len=nwords)

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
#     visualize(sentence_de,5,10,"{}".format(flname))
#     cntr += 1

