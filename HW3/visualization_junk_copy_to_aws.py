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
parser.add_argument('--adadelta','-ada',action='store_true',help='Use Adadelta optimizer')
parser.add_argument('--learning_rate','-lr',type=float,default=0.01,help='set learning rate.')
parser.add_argument('--rho','-r',type=float,default=0.95,help='rho for Adadelta optimizer')
parser.add_argument('--weight_decay','-wd',type=float,default=0.0,help='Weight decay constant for optimizer')
parser.add_argument('--accuracy','-acc',action='store_true',help='Calculate accuracy during training loop.')

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

sos_token = EN.vocab.stoi["<s>"]
eos_token = EN.vocab.stoi["</s>"]
pad_token = EN.vocab.stoi["<pad>"]

from collections import OrderedDict

class CandList():
    def __init__(self,n_layers,hidden_dim,n_de,beamsz=100):
        self.beamsz = beamsz
        self.hiddens = (torch.zeros(n_layers, 1, hidden_dim).cuda(),torch.zeros(n_layers, 1, hidden_dim).cuda())
        # hidden tensors (initially beamsz 1, later beamsz true beamsz)
        self.attentions = None
        # attention matrices-- we will concatenate along dimension 1. beamsz,n_en,n_de
        self.wordlist = None
        # wordlist will have dimension beamsz,iter
        self.probs = torch.zeros(1).cuda()
        # vector of probabilities, length beamsz
        self.firstloop = True
    def get_prev(self):
        if self.firstloop:
            return Variable(torch.cuda.LongTensor([sos_token]))
        else:
            return Variable(self.wordlist[:,-1])
    def get_hiddens(self):
        return (Variable(self.hiddens[0]),Variable(self.hiddens[1]))
    def update_beam(self,newlogprobs): # newlogprobs is beamsz,len(EN.vocab)
        newlogprobs = newlogprobs.data
        newlogprobs += self.probs.unsqueeze(1) # beamsz,len(EN.vocab)
        newlogprobs = newlogprobs.view(-1) # flatten to beamsz*len(EN.vocab) (search across all beams)
        sorte,indices = torch.topk(newlogprobs,self.beamsz) 
        # sorte and indices are beamsz. sorte contains probs, indices represent english word indices
        self.probs = sorte
        self.oldbeamindices = indices / len(EN.vocab)
        currbeam = indices % len(EN.vocab) # beamsz
        self.update_wordlist(currbeam)
    def update_wordlist(self,currbeam):
        # currbeam is beamsz vector of english word numbers
        currbeam = currbeam.unsqueeze(1)
        if self.firstloop:
            self.wordlist = currbeam
        else:
            shuffled = self.wordlist[self.oldbeamindices]
            self.wordlist = torch.cat([shuffled,currbeam],1)
        # self.wordlist is now beamsz,iter+1
    def update_hiddens(self,h,c):
        # no need to save old hidden states
        if self.firstloop:
            # see https://discuss.pytorch.org/t/initial-state-of-rnn-is-not-contiguous/4615
            h = h.expand(-1,self.beamsz,-1).contiguous()
            c = c.expand(-1,self.beamsz,-1).contiguous()
        # dimensions are n_layers*n_directions,beamsz,hiddensz
        self.hiddens = (h.data,c.data)
    def update_attentions(self,attn):
        attn = attn.unsqueeze(1)
        if self.firstloop:
            # attn is 1,1,n_de
            self.attentions = attn.data.expand(self.beamsz,-1,-1)
        else:
            # attn is beamsz,1,n_de
            unshuffled = torch.cat([self.attentions,attn.data],1)
            self.attentions = unshuffled[self.oldbeamindices]

class AttnNetwork(nn.Module):
    def __init__(self, word_dim=300, n_layers=1, hidden_dim=500, word2vec=False,
                vocab_layer_size=500, LSTM_dropout=0.0, vocab_layer_dropout=0.0, 
                 weight_tying=False, bidirectional=False, attn_type="soft"):
        super(AttnNetwork, self).__init__()
        self.attn_type = attn_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_layer_dim = (vocab_layer_size,word_dim)[weight_tying == True]
        self.directions = (1,2)[bidirectional == True]
         # LSTM initialization params: inputsz,hiddensz,n_layers,bias,batch_first,bidirectional
        self.encoder = nn.LSTM(word_dim, hidden_dim, num_layers = n_layers, batch_first = True, dropout=LSTM_dropout, bidirectional=bidirectional)
        self.decoder = nn.LSTM(word_dim, hidden_dim, num_layers = n_layers, batch_first = True, dropout=LSTM_dropout)
        self.embedding_de = nn.Embedding(len(DE.vocab), word_dim)
        self.embedding_en = nn.Embedding(len(EN.vocab), word_dim)
        if bidirectional:
            self.dim_reduce = nn.Linear(hidden_dim*2,hidden_dim)
        if word2vec:
            self.embedding_de.weight.data.copy_(DE.vocab.vectors)
            self.embedding_en.weight.data.copy_(EN.vocab.vectors)
        # vocab layer will combine dec hidden state with context vector, and then project out into vocab space 
        self.vocab_layer = nn.Sequential(OrderedDict([
            ('h2e',nn.Linear(hidden_dim*(self.directions+1), self.vocab_layer_dim)),
            ('tanh',nn.Tanh()),
            ('drp',nn.Dropout(vocab_layer_dropout)),
            ('e2v',nn.Linear(self.vocab_layer_dim, len(EN.vocab))),
            ('lsft',nn.LogSoftmax(dim=-1))
        ]))
        if weight_tying:
            self.vocab_layer.e2v.weight.data.copy_(self.embedding_en.weight.data)
        # baseline reward, which we initialize with log 1/V
        self.baseline = Variable(torch.cuda.FloatTensor([np.log(1/len(EN.vocab))]))
        # self.baseline = Variable(torch.zeros(1).fill_(np.log(1/len(EN.vocab))).cuda()) # yoon's way
    def forward(self, x_de, x_en, update_baseline=True):
        bs = x_de.size(0)
        # x_de is bs,n_de. x_en is bs,n_en
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en) # bs,n_en,word_dim
        h0_enc = torch.zeros(self.n_layers*self.directions, bs, self.hidden_dim).cuda()
        c0_enc = torch.zeros(self.n_layers*self.directions, bs, self.hidden_dim).cuda()
        h0_dec = torch.zeros(self.n_layers, bs, self.hidden_dim).cuda()
        c0_dec = torch.zeros(self.n_layers, bs, self.hidden_dim).cuda()
        # hidden vars have dimension n_layers*n_directions,bs,hiddensz
        enc_h, _ = self.encoder(emb_de, (Variable(h0_enc), Variable(c0_enc)))
        # enc_h is bs,n_de,hiddensz*n_directions. ordering is different from last week because batch_first=True
        dec_h, _ = self.decoder(emb_en, (Variable(h0_dec), Variable(c0_dec)))
        # dec_h is bs,n_en,hidden_size
        # we've gotten our encoder/decoder hidden states so we are ready to do attention
        # first let's get all our scores, which we can do easily since we are using dot-prod attention
        if self.directions == 2:
            scores = torch.bmm(self.dim_reduce(enc_h), dec_h.transpose(1,2))
            # TODO: any easier ways to reduce dimension?
        else:
            scores = torch.bmm(enc_h, dec_h.transpose(1,2))
        # (bs,n_de,hiddensz*n_directions) * (bs,hiddensz*n_directions,n_en) = (bs,n_de,n_en)
        loss = 0
        avg_reward = 0
        scores[(x_de == pad_token).unsqueeze(2).expand(scores.size())] = -math.inf # binary mask
        attn_dist = F.softmax(scores,dim=1) # bs,n_de,n_en
        # hard attn requires stacking to fit into torch.distributions.Categorical
        context = torch.bmm(attn_dist.transpose(2,1), enc_h)
        # (bs,n_en,n_de) * (bs,n_de,hiddensz) = (bs,n_en,hiddensz)
        pred = self.vocab_layer(torch.cat([dec_h,context],2)) # bs,n_en,len(EN.vocab)
        pred = pred[:,:-1,:] # alignment
        y = x_en[:,1:]
        reward = torch.gather(pred,2,y.unsqueeze(2)) # bs,n_en,1
        # reward[i,j,1] = input[i,j,y[i,j]]
        no_pad = (y != pad_token)
        reward = reward.squeeze(2)[no_pad]
        loss -= reward.sum() / no_pad.data.sum()
        avg_reward = -loss.data[0]
        # hard attention baseline and reinforce stuff causing me trouble
        return loss, 0, avg_reward
    # predict with greedy decoding and teacher forcing
    def predict(self, x_de, x_en):
        bs = x_de.size(0)
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en) # bs,n_en,word_dim
        h_enc = Variable(torch.zeros(self.n_layers*self.directions, bs, self.hidden_dim).cuda())
        c_enc = Variable(torch.zeros(self.n_layers*self.directions, bs, self.hidden_dim).cuda())
        h_dec = Variable(torch.zeros(self.n_layers, bs, self.hidden_dim).cuda())
        c_dec = Variable(torch.zeros(self.n_layers, bs, self.hidden_dim).cuda())
        enc_h, _ = self.encoder(emb_de, (h_enc, c_enc)) # (bs,n_de,hiddensz*2)
        dec_h, _ = self.decoder(emb_en, (h_dec, c_dec)) # (bs,n_en,hiddensz)
        # all the same. enc_h is bs,n_de,hiddensz*n_directions. h and c are both n_layers*n_directions,bs,hiddensz
        if self.directions == 2:
            scores = torch.bmm(self.dim_reduce(enc_h), dec_h.transpose(1,2))
        else:
            scores = torch.bmm(enc_h, dec_h.transpose(1,2))
        # (bs,n_de,hiddensz) * (bs,hiddensz,n_en) = (bs,n_de,n_en)
        scores[(x_de == pad_token).unsqueeze(2).expand(scores.size())] = -math.inf # binary mask
        attn_dist = F.softmax(scores,dim=1) # bs,n_de,n_en
        context = torch.bmm(attn_dist.transpose(2,1),enc_h)
        # (bs,n_en,n_de) * (bs,n_de,hiddensz*ndirections) = (bs,n_en,hiddensz*ndirections)
        pred = self.vocab_layer(torch.cat([dec_h,context],2)) # bs,n_en,len(EN.vocab)
        pred = pred[:,:-1,:] # alignment
        _, tokens = pred.max(2) # bs,n_en-1
        sauce = Variable(torch.cuda.LongTensor([[sos_token]]*bs)) # bs
        return torch.cat([sauce,tokens],1), attn_dist
    # Singleton batch with BSO
    def predict2(self, x_de, beamsz, gen_len):
        emb_de = self.embedding_de(x_de) # "batch size",n_de,word_dim, but "batch size" is 1 in this case!
        h0_enc = Variable(torch.zeros(self.n_layers*self.directions, 1, self.hidden_dim).cuda())
        c0_enc = Variable(torch.zeros(self.n_layers*self.directions, 1, self.hidden_dim).cuda())
        h0_dec = Variable(torch.zeros(self.n_layers, 1, self.hidden_dim).cuda())
        c0_dec = Variable(torch.zeros(self.n_layers, 1, self.hidden_dim).cuda())
        enc_h, _ = self.encoder(emb_de, (h0_enc, c0_enc))
        # since enc batch size=1, enc_h is 1,n_de,hiddensz*n_directions
        masterheap = CandList(self.n_layers,self.hidden_dim,enc_h.size(1),beamsz)
        # in the following loop, beamsz is length 1 for first iteration, length true beamsz (100) afterward
        for i in range(gen_len):
            prev = masterheap.get_prev() # beamsz
            emb_t = self.embedding_en(prev) # embed the last thing we generated. beamsz,word_dim
            enc_h_expand = enc_h.expand(prev.size(0),-1,-1) # beamsz,n_de,hiddensz
            
            h, c = masterheap.get_hiddens() # (n_layers,beamsz,hiddensz),(n_layers,beamsz,hiddensz)
            dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c)) # dec_h is beamsz,1,hiddensz (batch_first=True)
            if self.directions == 2:
                scores = torch.bmm(self.dim_reduce(enc_h_expand), dec_h.transpose(1,2)).squeeze(2)
            else:
                scores = torch.bmm(enc_h_expand, dec_h.transpose(1,2)).squeeze(2)
            # (beamsz,n_de,hiddensz) * (beamsz,hiddensz,1) = (beamsz,n_de,1). squeeze to beamsz,n_de
            scores[(x_de == pad_token)] = -math.inf # binary mask
            attn_dist = F.softmax(scores,dim=1)
            if self.attn_type == "hard":
                _, argmax = attn_dist.max(1) # beamsz for each batch, select most likely german word to pay attention to
                one_hot = Variable(torch.zeros_like(attn_dist.data).scatter_(-1, argmax.data.unsqueeze(1), 1).cuda())
                context = torch.bmm(one_hot.unsqueeze(1), enc_h_expand).squeeze(1)
            else:
                context = torch.bmm(attn_dist.unsqueeze(1), enc_h_expand).squeeze(1)
            # the difference btwn hard and soft is just whether we use a one_hot or a distribution
            # context is beamsz,hiddensz*n_directions
            pred = self.vocab_layer(torch.cat([dec_h.squeeze(1), context], 1)) # beamsz,len(EN.vocab)
            # TODO: set the columns corresponding to <pad>,<unk>,</s>,etc to 0
            masterheap.update_beam(pred)
            masterheap.update_hiddens(h,c)
            masterheap.update_attentions(attn_dist)
            masterheap.firstloop = False
        return masterheap.probs,masterheap.wordlist,masterheap.attentions


model = AttnNetwork(word_dim=args.embedding_dims, n_layers=4, hidden_dim=1000, word2vec=args.word2vec,
                        vocab_layer_size=300, LSTM_dropout=0.3, vocab_layer_dropout=0.3, 
                        weight_tying=args.weight_tying, bidirectional=True, attn_type=attn_type)



model.load_state_dict(torch.load('../../models/HW3/wwt.pkl'))

model.cuda()



def visualize(sentence_de,bs,nwords,flname): # attns = (SentLen_EN)x(SentLen_DE), sentence_de = ["German_1",...,"German_(SentLen_DE)"]
    ls = [[DE.vocab.stoi[w] for w in sentence_de.split(' ')]]
    _,wordlist,attns = model.predict2(Variable(torch.cuda.LongTensor(ls)),beamsz=bs,gen_len=nwords)
    topbeams = []
    for j in range(bs):
        topbeams.append([EN.vocab.itos[w] for w in wordlist[0]])
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    # print(type(attns.cpu))
    attnscpu = attns.cpu().numpy()
    cax = ax.matshow(attnscpu[0], cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticklabels(['']+sentence_de.split(' '),rotation=90)
    ax.set_yticklabels(['']+topbeams[0])
    print(['']+sentence_de.split(' '))
    print(['']+topbeams[0])
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.savefig("{}.png".format(flname))

list_of_german_sentences = [[""]]

cntr = 0
for sentence_de in list_of_german_sentences:
    flname = "plot_"+"{}".format(cntr)
    visualize(sentence_de,5,10,"{}".format(flname))
    cntr += 1