import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import math # for infinity
from __main__ import *
from helpers import lstm_hidden, unpackage_hidden
# I use EN,DE,BATCH_SIZE,MAX_LEN,pad_token,sos_token,eos_token,word2vec

#######################################
#             ATTENTION               #
#  This version assumes that you are  #
#  not using hard attention and that  #
#  you are using teacher forcing.     #
#######################################


# If we have beamsz 100 and we only go 3 steps, we're guaranteed to have 100 unique trigrams
# This is written so that, at any time, hiddens/attentions/wordlist/probs all align with each other across the beamsz dimension
# - We could've enforced this better by packaging together each beam member in an object, but we don't
# philosophy: we're going to take the dirty variables and reorder them nicely all in here and store them as tensors
# the inputs to these methods could have beamsz = 1 or beamsz = true beamsz (100)
class CandList():
    def __init__(self,n_de,initHidden,beamsz=100):
        self.beamsz = beamsz
        self.hiddens = unpackage_hidden(initHidden) # would this break if i called self.update_hidden(initHidden)?
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
        try:
            res = tuple( Variable(x) for x in self.hiddens )
        except TypeError:
            res = Variable(self.hiddens)
        return res
        #return (Variable(self.hiddens[0]),Variable(self.hiddens[1]))
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
    def update_hiddens(self,hidd):
        # no need to save old hidden states
        if self.firstloop:
            try:
                hidd = ( x.expand(-1,self.beamsz,-1).contiguous() for x in hidd )
            except TypeError:
                hidd = hidd.expand(-1,self.beamsz,-1).contiguous()
            # see https://discuss.pytorch.org/t/initial-state-of-rnn-is-not-contiguous/4615
            #h = h.expand(-1,self.beamsz,-1).contiguous()
            #c = c.expand(-1,self.beamsz,-1).contiguous()
        # dimensions are n_layers*n_directions,beamsz,hiddensz
        #self.hiddens = (h.data,c.data)
        self.hiddens = unpackage_hidden(hidd)
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
    def initEnc(self,batch_size):
        return (Variable(torch.zeros(self.n_layers*self.directions,batch_size,self.hidden_dim).cuda()), 
                Variable(torch.zeros(self.n_layers*self.directions,batch_size,self.hidden_dim).cuda()))
    def initDec(self,batch_size):
        return (Variable(torch.zeros(self.n_layers,batch_size,self.hidden_dim).cuda()), 
                Variable(torch.zeros(self.n_layers,batch_size,self.hidden_dim).cuda()))
    def forward(self, x_de, x_en, update_baseline=True):
        bs = x_de.size(0)
        # x_de is bs,n_de. x_en is bs,n_en
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en) # bs,n_en,word_dim
        # hidden vars have dimension n_layers*n_directions,bs,hiddensz
        enc_h, _ = self.encoder(emb_de, self.initEnc(bs))
        # enc_h is bs,n_de,hiddensz*n_directions. ordering is different from last week because batch_first=True
        dec_h, _ = self.decoder(emb_en, self.initDec(bs))
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
        enc_h, _ = self.encoder(emb_de, self.initEnc(bs)) # (bs,n_de,hiddensz*2)
        dec_h, _ = self.decoder(emb_en, self.initDec(bs)) # (bs,n_en,hiddensz)
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
        # pred[:,:,[unk_token,pad_token]] = -math.inf # TODO: testing this out kill pad unk
        pred = pred[:,:-1,:] # alignment
        _, tokens = pred.max(2) # bs,n_en-1
        sauce = Variable(torch.cuda.LongTensor([[sos_token]]*bs)) # bs
        return torch.cat([sauce,tokens],1), attn_dist
    # Singleton batch with BSO
    def predict2(self, x_de, beamsz, gen_len):
        emb_de = self.embedding_de(x_de) # "batch size",n_de,word_dim, but "batch size" is 1 in this case!
        enc_h, _ = self.encoder(emb_de, self.initEnc(1))
        # since enc batch size=1, enc_h is 1,n_de,hiddensz*n_directions
        masterheap = CandList(enc_h.size(1),self.initDec(1),beamsz)
        # in the following loop, beamsz is length 1 for first iteration, length true beamsz (100) afterward
        for i in range(gen_len):
            prev = masterheap.get_prev() # beamsz
            emb_t = self.embedding_en(prev) # embed the last thing we generated. beamsz,word_dim
            enc_h_expand = enc_h.expand(prev.size(0),-1,-1) # beamsz,n_de,hiddensz
            #
            hidd = masterheap.get_hiddens() # (n_layers,beamsz,hiddensz),(n_layers,beamsz,hiddensz)
            dec_h, hidd = self.decoder(emb_t.unsqueeze(1), hidd) # dec_h is beamsz,1,hiddensz (batch_first=True)
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
            # pred[:,:,[unk_token,pad_token]] = -inf # TODO: testing this out kill pad unk
            masterheap.update_beam(pred)
            masterheap.update_hiddens(hidd)
            masterheap.update_attentions(attn_dist)
            masterheap.firstloop = False
        return masterheap.probs,masterheap.wordlist,masterheap.attentions

class AttnGRU(nn.Module):
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
        self.encoder = nn.GRU(word_dim, hidden_dim, num_layers = n_layers, batch_first = True, dropout=LSTM_dropout, bidirectional=bidirectional)
        self.decoder = nn.GRU(word_dim, hidden_dim, num_layers = n_layers, batch_first = True, dropout=LSTM_dropout)
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
    def initEnc(self,batch_size):
        return Variable(torch.zeros(self.n_layers*self.directions,batch_size,self.hidden_dim).cuda())
    def initDec(self,batch_size):
        return Variable(torch.zeros(self.n_layers,batch_size,self.hidden_dim).cuda())
    def forward(self, x_de, x_en, update_baseline=True):
        bs = x_de.size(0)
        # x_de is bs,n_de. x_en is bs,n_en
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en) # bs,n_en,word_dim
        # hidden vars have dimension n_layers*n_directions,bs,hiddensz
        enc_h, _ = self.encoder(emb_de, self.initEnc(bs))
        # enc_h is bs,n_de,hiddensz*n_directions. ordering is different from last week because batch_first=True
        dec_h, _ = self.decoder(emb_en, self.initDec(bs))
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
        enc_h, _ = self.encoder(emb_de, self.initEnc(bs)) # (bs,n_de,hiddensz*2)
        dec_h, _ = self.decoder(emb_en, self.initDec(bs)) # (bs,n_en,hiddensz)
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
        enc_h, _ = self.encoder(emb_de, self.initEnc(1))
        # since enc batch size=1, enc_h is 1,n_de,hiddensz*n_directions
        masterheap = CandList(self.n_layers,self.hidden_dim,enc_h.size(1),beamsz)
        # in the following loop, beamsz is length 1 for first iteration, length true beamsz (100) afterward
        for i in range(gen_len):
            prev = masterheap.get_prev() # beamsz
            emb_t = self.embedding_en(prev) # embed the last thing we generated. beamsz,word_dim
            enc_h_expand = enc_h.expand(prev.size(0),-1,-1) # beamsz,n_de,hiddensz
            
            h, c = masterheap.get_hiddens() # (n_layers,beamsz,hiddensz),(n_layers,beamsz,hiddensz)
            dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), h) # dec_h is beamsz,1,hiddensz (batch_first=True)
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

class S2S(nn.Module):
    def __init__(self, word_dim=300, n_layers=1, hidden_dim=500, word2vec=False,
                vocab_layer_size=500, LSTM_dropout=0.0, vocab_layer_dropout=0.0, 
                weight_tying=False, bidirectional=False):
        super(S2S, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_layer_dim = (vocab_layer_size,word_dim)[weight_tying == True]
        self.directions = (1,2)[bidirectional == True]
        # LSTM initialization params: inputsz,hiddensz,n_layers,bias,batch_first,bidirectional
        self.encoder = nn.LSTM(word_dim, hidden_dim, n_layers, batch_first = True, dropout=LSTM_dropout, bidirectional=bidirectional)
        self.decoder = nn.LSTM(word_dim, hidden_dim, n_layers*self.directions, batch_first = True, dropout=LSTM_dropout)
        self.embedding_de = nn.Embedding(len(DE.vocab), word_dim)
        self.embedding_en = nn.Embedding(len(EN.vocab), word_dim)
        if word2vec:
            self.embedding_de.weight.data.copy_(DE.vocab.vectors)
            self.embedding_en.weight.data.copy_(EN.vocab.vectors)
        # vocab layer will project dec hidden state out into vocab space 
        self.vocab_layer = nn.Sequential(OrderedDict([
            ('h2e',nn.Linear(hidden_dim,self.vocab_layer_dim)),
            ('tanh',nn.Tanh()),
            ('drp',nn.Dropout(vocab_layer_dropout)),
            ('e2v',nn.Linear(self.vocab_layer_dim,len(EN.vocab))),
            ('lsft',nn.LogSoftmax(dim=-1))
        ]))
        if weight_tying:
            self.vocab_layer.e2v.weight.data.copy_(self.embedding_en.weight.data)
        self.baseline = Variable(torch.cuda.FloatTensor([np.log(1/len(EN.vocab))])) # just to be consistent
    def initEnc(self,batch_size):
        return (Variable(torch.zeros(self.n_layers*self.directions,batch_size,self.hidden_dim).cuda()), 
                Variable(torch.zeros(self.n_layers*self.directions,batch_size,self.hidden_dim).cuda()))
    def forward(self, x_de, x_en):
        bs = x_de.size(0)
        # x_de is bs,n_de. x_en is bs,n_en
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en) # bs,n_en,word_dim
        # hidden vars have dimension n_layers*n_directions,bs,hiddensz
        enc_h, (h,c) = self.encoder(emb_de, self.initEnc(bs))
        # enc_h is bs,n_de,hiddensz*n_directions. ordering is different from last week because batch_first=True
        dec_h, _ = self.decoder(emb_en, (h,c))
        # dec_h is bs,n_en,hidden_size
        pred = self.vocab_layer(dec_h) # bs,n_en,len(EN.vocab)
        pred = pred[:,:-1,:] # alignment
        y = x_en[:,1:]
        reward = torch.gather(pred,2,y.unsqueeze(2))
        # reward[i,j,1] = pred[i,j,y[i,j]]
        no_pad = (y != pad_token)
        reward = reward.squeeze(2)[no_pad] # less than bs,n_en
        loss = -reward.sum() / no_pad.data.sum()
        return loss, 0, -loss.data[0] # passing back things just to be consistent
    # predict with greedy decoding and teacher forcing
    def predict(self, x_de, x_en):
        bs = x_de.size(0)
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en)
        enc_h, (h,c) = self.encoder(emb_de, self.initEnc(bs))
        dec_h, _ = self.decoder(emb_en, (h,c))
        # all the same. enc_h is bs,n_de,hiddensz*n_directions. h and c are both n_layers*n_directions,bs,hiddensz
        pred = self.vocab_layer(dec_h) # bs,n_en,len(EN.vocab)
        pred = pred[:,:-1,:] # alignment
        _, tokens = pred.max(2) # bs,n_en
        sauce = Variable(torch.cuda.LongTensor([[sos_token]]*bs)) # bs
        return torch.cat([sauce,tokens],1), 0 # no attention to return
    # Singleton batch with BSO
    def predict2(self, x_de, beamsz, gen_len):
        emb_de = self.embedding_de(x_de) # "batch size",n_de,word_dim, but "batch size" is 1 in this case!
        enc_h, (h, c) = self.encoder(emb_de, self.initEnc(1))
        if self.directions == 2:
            h = self.dim_reduce(h.transpose(2,0)).transpose(2,0).contiguous() # nlayers*2,bs,hiddensz to nlayers,bs,hiddensz
            c = self.dim_reduce(c.transpose(2,0)).transpose(2,0).contiguous() # this is hacky, and i don't like doing it!
        # since enc batch size=1, enc_h is 1,n_de,hiddensz*n_directions
        masterheap = CandList(enc_h.size(1),(h,c),beamsz)
        masterheap.update_hiddens((h,c)) # TODO: this extraneous call could be eliminated if __init__ called self.update_hiddens
        # in the following loop, beamsz is length 1 for first iteration, length true beamsz (100) afterward
        for i in range(gen_len):
            prev = masterheap.get_prev() # beamsz
            emb_t = self.embedding_en(prev) # embed the last thing we generated. beamsz,word_dim
            enc_h_expand = enc_h.expand(prev.size(0),-1,-1) # beamsz,n_de,hiddensz
            h, c = masterheap.get_hiddens() # (n_layers,beamsz,hiddensz),(n_layers,beamsz,hiddensz)
            dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c)) # dec_h is beamsz,1,hiddensz (batch_first=True)
            pred = self.vocab_layer(dec_h.squeeze(1)) # beamsz,len(EN.vocab)
            # TODO: set the columns corresponding to <pad>,<unk>,</s>,etc to 0
            masterheap.update_beam(pred)
            masterheap.update_hiddens(h,c)
            masterheap.update_attentions(attn_dist)
            masterheap.firstloop = False
        return masterheap.probs,masterheap.wordlist,masterheap.attentions
