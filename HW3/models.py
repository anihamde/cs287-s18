import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import math # for infinity
from __main__ import *
from helpers import lstm_hidden, unpackage_hidden, freeze_model
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
        if type(self.hiddens) == torch.cuda.FloatTensor:
            res = Variable(self.hiddens)
        else:
            res = tuple( Variable(x) for x in self.hiddens )
        #try:
        #    res = tuple( Variable(x) for x in self.hiddens )
        #except TypeError:
        #    res = Variable(self.hiddens)
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
            if type(hidd) == Variable:
                hidd = hidd.expand(-1,self.beamsz,-1).contiguous()
            else:
                hidd = ( x.expand(-1,self.beamsz,-1).contiguous() for x in hidd )
            #try:
            #    hidd = ( x.expand(-1,self.beamsz,-1).contiguous() for x in hidd )
            #except TypeError:
            #    hidd = hidd.expand(-1,self.beamsz,-1).contiguous()
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
        return loss, 0, avg_reward, pred
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

class AttnCNN(nn.Module):
    def __init__(self, word_dim=300, n_layers=1, hidden_dim=500, word2vec=False,
                vocab_layer_size=500, LSTM_dropout=0.0, vocab_layer_dropout=0.0, 
                 weight_tying=False, bidirectional=False, attn_type="soft"):
        super(AttnCNN, self).__init__()
        self.attn_type = attn_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_layer_dim = (vocab_layer_size,word_dim)[weight_tying == True]
        self.directions = (1,2)[bidirectional == True]
         # LSTM initialization params: inputsz,hiddensz,n_layers,bias,batch_first,bidirectional
        self.conv3_enc = nn.Sequential(nn.Conv2d(word_dim, self.hidden_dim,kernel_size=(3,1),padding=(1,0)),nn.Tanh())
        self.conv3_dec = nn.Sequential(nn.Conv2d(word_dim, self.hidden_dim,kernel_size=(3,1),padding=(1,0)),nn.Tanh())
        if self.n_layers > 1:
            self.c3_seq_enc = nn.Sequential(*[ a for b in tuple( tuple((nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=(3,1),padding=(1,0)),nn.Tanh())) for _ in range(1,self.n_layers) ) for a in b ])
            self.c3_seq_dec = nn.Sequential(*[ a for b in tuple( tuple((nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=(3,1),padding=(1,0)),nn.Tanh())) for _ in range(1,self.n_layers) ) for a in b ])            
        self.embedding_de = nn.Embedding(len(DE.vocab), word_dim)
        self.embedding_en = nn.Embedding(len(EN.vocab), word_dim)
        #if hidden_dim != (n_featmaps1+n_featmaps2):
        #    self.dim_reduce = nn.Linear(n_featmaps1+n_featmaps2,hidden_dim)
        if word2vec:
            self.embedding_de.weight.data.copy_(DE.vocab.vectors)
            self.embedding_en.weight.data.copy_(EN.vocab.vectors)
        # vocab layer will combine dec hidden state with context vector, and then project out into vocab space 
        self.vocab_layer = nn.Sequential(OrderedDict([
            ('h2e',nn.Linear(self.hidden_dim+self.hidden_dim, self.vocab_layer_dim)),
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
    def encoder(self, emb_de, dummy):
        enc_h = emb_de.unsqueeze(2)
        enc_h = enc_h.permute(0,3,1,2)
        enc_h = self.conv3_enc(enc_h)
        if self.n_layers > 1:
            enc_h = self.c3_seq_enc(enc_h)
        enc_h = enc_h.squeeze(3)
        enc_h = enc_h.permute(0,2,1)
        return enc_h, "poop"
    def decoder(self, emb_en, dummy):
        dec_h = emb_en.unsqueeze(2)
        dec_h = dec_h.permute(0,3,1,2)
        dec_h = self.conv3_enc(dec_h)
        if self.n_layers > 1:
            dec_h = self.c3_seq_dec(dec_h)
        dec_h = dec_h.squeeze(3)
        dec_h = dec_h.permute(0,2,1)
        return dec_h, "poop"
    def forward(self, x_de, x_en, update_baseline=True):
        bs = x_de.size(0)
        # x_de is bs,n_de. x_en is bs,n_en
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en) # bs,n_en,word_dim
        # hidden vars have dimension n_layers*n_directions,bs,hiddensz
        # enc_h is bs,n_de,hiddensz*n_directions. ordering is different from last week because batch_first=True
        enc_h, _ = self.encoder(emb_de, self.initEnc(bs))
        dec_h, _ = self.decoder(emb_en, self.initDec(bs))
        # dec_h is bs,n_en,hidden_size
        # we've gotten our encoder/decoder hidden states so we are ready to do attention
        # first let's get all our scores, which we can do easily since we are using dot-prod attention
        #if self.hidden_dim != (self.n_featmaps1+self.n_featmaps2):
        #    scores = torch.bmm(self.dim_reduce(enc_h), dec_h.transpose(1,2))
            # TODO: any easier ways to reduce dimension?
        if True:
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
        return loss, 0, avg_reward, pred
    # predict with greedy decoding and teacher forcing
    def predict(self, x_de, x_en):
        bs = x_de.size(0)
        enc_h, _ = self.encoder(emb_de, self.initEnc(bs))
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
        super(AttnGRU, self).__init__()
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
        return loss, 0, avg_reward, pred
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
        masterheap = CandList(enc_h.size(1),self.initDec(1),beamsz)
        # in the following loop, beamsz is length 1 for first iteration, length true beamsz (100) afterward
        for i in range(gen_len):
            prev = masterheap.get_prev() # beamsz
            emb_t = self.embedding_en(prev) # embed the last thing we generated. beamsz,word_dim
            enc_h_expand = enc_h.expand(prev.size(0),-1,-1) # beamsz,n_de,hiddensz
            
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
            # TODO: set the columns corresponding to <pad>,<unk>,</s>,etc to 0
            masterheap.update_beam(pred)
            masterheap.update_hiddens(hidd)
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
        loss = 0
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
        return loss, 0, -loss.data[0], pred # passing back things just to be consistent
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
        # since enc batch size=1, enc_h is 1,n_de,hiddensz*n_directions
        masterheap = CandList(enc_h.size(1),(h,c),beamsz)
        #masterheap.update_hiddens((h,c)) # TODO: this extraneous call could be eliminated if __init__ called self.update_hiddens
        # in the following loop, beamsz is length 1 for first iteration, length true beamsz (100) afterward
        for i in range(gen_len):
            prev = masterheap.get_prev() # beamsz
            emb_t = self.embedding_en(prev) # embed the last thing we generated. beamsz,word_dim
            h, c = masterheap.get_hiddens() # (n_layers,beamsz,hiddensz),(n_layers,beamsz,hiddensz)
            dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c)) # dec_h is beamsz,1,hiddensz (batch_first=True)
            pred = self.vocab_layer(dec_h.squeeze(1)) # beamsz,len(EN.vocab)
            # TODO: set the columns corresponding to <pad>,<unk>,</s>,etc to 0
            masterheap.update_beam(pred)
            masterheap.update_hiddens((h,c))
            masterheap.update_attentions(attn_dist)
            masterheap.firstloop = False
        return masterheap.probs,masterheap.wordlist,masterheap.attentions

class Alpha(nn.Module):
    def __init__(self, models_tuple, embedding_features=300, n_featmaps1=200, n_featmaps2=100, linear_size=300, dropout_rate=0.5, word2vec=False, freeze_models=False):
        super(Alpha, self).__init__()
        if freeze_models:
            self.members = tuple( freeze_model(x) for x in models_tuple )
        else:
            self.members = models_tuple
        self.member_count = len(models_tuple)
        self.embedding_dims = (embedding_features, 300)[word2vec == True]
        self.n_featmaps1 = n_featmaps1
        self.n_featmaps2 = n_featmaps2
        self.embedding = nn.Embedding(len(DE.vocab), self.embedding_dims)
        if word2vec:
            self.embedding.weight.data.copy_(DE.vocab.vectors)
        self.conv3 = nn.Conv2d(300,self.n_featmaps1,kernel_size=(3,1),padding=(1,0))
        self.conv5 = nn.Conv2d(300,self.n_featmaps2,kernel_size=(5,1),padding=(2,0))
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden = nn.Linear(n_featmaps1+n_featmaps2,linear_size)
        self.output = nn.Linear(linear_size,self.member_count)
        # vocab layer will combine dec hidden state with context vector, and then project out into vocab space 
        self.baseline = Variable(torch.cuda.FloatTensor([np.log(1/len(EN.vocab))])) # just to be consistent
    def initEnc(self,batch_size):
        return (Variable(torch.zeros(self.n_layers*self.directions,batch_size,self.hidden_dim).cuda()), 
                Variable(torch.zeros(self.n_layers*self.directions,batch_size,self.hidden_dim).cuda()))
    def initDec(self,batch_size):
        return (Variable(torch.zeros(self.n_layers,batch_size,self.hidden_dim).cuda()), 
                Variable(torch.zeros(self.n_layers,batch_size,self.hidden_dim).cuda()))
    def get_alpha(self, x_de):
        bs = x_de.size(0)
        embeds = self.embedding(x_de) # bs,n_de,word_dim
        out = embeds.unsqueeze(2)
        out = out.permute(0,3,1,2) # bs,word_dim,n_de,1
        fw3 = self.conv3(out) # bs,n_featmaps1,n_de,1
        fw5 = self.conv5(out) # bs,n_featmaps2,n_de,1
        out = torch.cat([fw3,fw5],dim=1) # bs,n_featmaps1+n_featmaps2,n_de,1
        out = out.squeeze(-1) # bs,n_featmaps1+n_featmaps2,n_de
        out = self.maxpool(out) # bs,n_featmaps1+n_featmaps2,1
        out = out.squeeze(-1) # bs,n_featmaps1+n_featmaps2
        out = self.hidden(out) # bs, linear_size
        out = self.dropout(out)
        out = self.output(out) # bs, len(model_tuple)
        out = F.softmax(out,dim=1) # bs, len(model_tuple)
        out = out.unsqueeze(1) # bs, 1, len(model_tuple)
        out = out.unsqueeze(2) # bs, 1, 1, len(model_tuple)
        return out
        #
    def forward(self, x_de, x_en):
        loss = 0
        out = self.get_alpha(x_de)
        models_stack = torch.stack(tuple( x.forward(x_de,x_en)[3] for x in self.members ),dim=3) # bs,n_en,len(EN.vocab),len(models_tuple)
        out = models_stack * out
        pred = out.sum(3) # bs,n_en,len(EN.vocab) 
        y = x_en[:,1:]
        reward = torch.gather(pred,2,y.unsqueeze(2)) # bs,n_en,1
        no_pad = (y != pad_token)
        reward = reward.squeeze(2)[no_pad]
        loss -= reward.sum() / no_pad.data.sum()
        avg_reward = -loss.data[0]
        # hard attention baseline and reinforce stuff causing me trouble
        return loss, 0, avg_reward, pred
    def predict(self, x_de, x_en):
        out = self.get_alpha(x_de)
        models_stack = torch.stack(tuple( x.forward(x_de,x_en)[3] for x in self.members ),dim=3) # bs,n_en,len(EN.vocab),len(models_tuple)
        out = models_stack * out
        pred = out.sum(3) # bs,n_en,len(EN.vocab)
        # the below is literally copy pasted from previous predict fnctions
        _, tokens = pred.max(2) # bs,n_en-1
        sauce = Variable(torch.cuda.LongTensor([[sos_token]]*bs)) # bs
        return torch.cat([sauce,tokens],1), attn_dist
    def predict2(self,x_de,beamsz,gen_len):
        out = self.get_alpha(x_de)
        #
        r_dex = range(self.member_count)
        emb_de = tuple( self.members[i].embedding_de(x_de) for i in r_dex )
        enc_h  = tuple( self.members[i].encoder(emb_de[i],self.members[i].initEnc(1))[0] for i in r_dex )
        masterheaps = tuple( CandList(enc_h[i],self.members[i].initDec(1),beamsz) for i in r_dex )
        for _ in range(gen_len):
            prev  = tuple( heap.get_prev() for heap in masterheaps )
            emb_t = tuple( self.members[i].embedding_en(prev[i]) for i in r_dex )
            enc_h_expand = tuple( enc_h[i].expand(prev[i].size(0),-1,-1) for i in r_dex )
            hidd = tuple( heap.get_hiddens() for heap in masterheaps )
            hold = tuple( self.members[i].decoder(emb_t[i].unsqueeze(1),hidd[i]) for i in r_dex )
            dec_h, hidd = tuple(zip(*hold))
            scores = tuple( torch.bmm(self.members[i].dim_reduce(enc_h_expand[i]), dec_h[i].transpose(1,2)).squeeze(2) if self.members[i].directions == 2 else torch.bmm(enc_h_expand[i], dec_h[i].transpose(1,2)).squeeze(2) for i in r_dex )
            for i in r_dex:
                scores[i][(x_de == pad_token)] = -math.inf
            attn_dist = tuple( F.softmax(scores[i],dim=1) for i in r_dex )
            context = tuple( torch.bmm(attn_dist[i].unsqueeze(1),enc_h_expand[i]).squeeze(1) for i in r_dex )
            pred = tuple( self.members[i].vocab_layer(torch.cat([dec_h[i].squeeze(1), context[i]], 1)) for i in r_dex )
            weighted_pred  = torch.stack(pred,dim=2) * out.squeeze(2)
            ensembled_pred = weighted_pred.sum(2)
            for i in r_dex:
                masterheaps[i].update_beam(ensembled_pred)
                masterheaps[i].update_hiddens(hidd[i])
                masterheaps[i].update_attentions(attn_dist[i])
                masterheaps[i].firstloop = False
        return "poop",masterheaps[0].wordlist,"morepoop"
        #return masterheap.probs,masterheap.wordlist,masterheap.attentions
        #
        #assert(0==1)
class Beta(nn.Module):
    def __init__(self, models_tuple, embedding_features=300, hidden_size=200, n_layers=2, linear_size=300, dropout_rate=0.5, bidirectional=False, word2vec=False, freeze_models=False):
        super(Beta, self).__init__()
        if freeze_models:
            self.members = tuple( freeze_model(x) for x in models_tuple )
        else:
            self.members = models_tuple
        self.member_count = len(models_tuple)
        self.embedding_dims = (embedding_features, 300)[word2vec == True]
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.linear_size = linear_size
        self.directions = (1,2)[bidirectional == True]
        self.embedding = nn.Embedding(len(DE.vocab), self.embedding_dims)
        if word2vec:
            self.embedding.weight.data.copy_(DE.vocab.vectors)
        self.lstm = nn.LSTM(self.embedding_dims, self.hidden_size, self.n_layers, dropout=dropout_rate, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size*self.directions, self.linear_size)#len(TEXT.vocab))        
        self.output = nn.Linear(self.linear_size,self.member_count)
        # vocab layer will combine dec hidden state with context vector, and then project out into vocab space 
        self.baseline = Variable(torch.cuda.FloatTensor([np.log(1/len(EN.vocab))])) # just to be consistent
    def initEnc(self,batch_size):
        return (Variable(torch.zeros(self.n_layers*self.directions,batch_size,self.hidden_size).cuda()), 
                Variable(torch.zeros(self.n_layers*self.directions,batch_size,self.hidden_size).cuda()))
    def initDec(self,batch_size):
        return (Variable(torch.zeros(self.n_layers,batch_size,self.hidden_size).cuda()), 
                Variable(torch.zeros(self.n_layers,batch_size,self.hidden_size).cuda()))
    def get_alpha(self, x_de):
        bs = x_de.size(0)
        embeds = self.embedding(x_de) # bs,n_de,word_dim
        out = self.dropout(embeds) # bs,n_de,word_dim
        out, hidden = self.lstm(out, self.initEnc(bs)) # n_de,bs,directions*hidden_dim
        out = out[:,-1,:] # bs,directions*hidden_dim
        out = self.dropout(out) # bs,directions*hidden_dim
        out = self.linear(out) # bs,linear_size
        out = self.output(out) # bs, len(model_tuple)
        out = F.softmax(out,dim=1) # bs, len(model_tuple)
        out = out.unsqueeze(1) # bs, 1, len(model_tuple)
        out = out.unsqueeze(2) # bs, 1, 1, len(model_tuple)
        return out
        #
    def forward(self, x_de, x_en):
        loss = 0
        out = self.get_alpha(x_de)
        models_stack = torch.stack(tuple( x.forward(x_de,x_en)[3] for x in self.members ),dim=3) # bs,n_en,len(EN.vocab),len(models_tuple)
        out = models_stack * out
        pred = out.sum(3) # bs,n_en,len(EN.vocab) 
        y = x_en[:,1:]
        reward = torch.gather(pred,2,y.unsqueeze(2)) # bs,n_en,1
        no_pad = (y != pad_token)
        reward = reward.squeeze(2)[no_pad]
        loss -= reward.sum() / no_pad.data.sum()
        avg_reward = -loss.data[0]
        # hard attention baseline and reinforce stuff causing me trouble
        return loss, 0, avg_reward, pred
    def predict(self, x_de, x_en):
        out = self.get_alpha(x_de)
        models_stack = torch.stack(tuple( x.forward(x_de,x_en)[3] for x in self.members ),dim=3) # bs,n_en,len(EN.vocab),len(models_tuple)
        out = models_stack * out
        pred = out.sum(3) # bs,n_en,len(EN.vocab)
        # the below is literally copy pasted from previous predict fnctions
        _, tokens = pred.max(2) # bs,n_en-1
        sauce = Variable(torch.cuda.LongTensor([[sos_token]]*bs)) # bs
        return torch.cat([sauce,tokens],1), attn_dist
    def predict2(self,x_de,beamsz,gen_len):
        out = self.get_alpha(x_de)
        #
        r_dex = range(self.member_count)
        emb_de = tuple( self.members[i].embedding_de(x_de) for i in r_dex )
        enc_h  = tuple( self.members[i].encoder(emb_de[i],self.members[i].initEnc(1))[0] for i in r_dex )
        masterheaps = tuple( CandList(enc_h[i],self.members[i].initDec(1),beamsz) for i in r_dex )
        for _ in range(gen_len):
            prev  = tuple( heap.get_prev() for heap in masterheaps )
            emb_t = tuple( self.members[i].embedding_en(prev[i]) for i in r_dex )
            enc_h_expand = tuple( enc_h[i].expand(prev[i].size(0),-1,-1) for i in r_dex )
            hidd = tuple( heap.get_hiddens() for heap in masterheaps )
            hold = tuple( self.members[i].decoder(emb_t[i].unsqueeze(1),hidd[i]) for i in r_dex )
            dec_h, hidd = tuple(zip(*hold))
            scores = tuple( torch.bmm(self.members[i].dim_reduce(enc_h_expand[i]), dec_h[i].transpose(1,2)).squeeze(2) if self.members[i].directions == 2 else torch.bmm(enc_h_expand[i], dec_h[i].transpose(1,2)).squeeze(2) for i in r_dex )
            for i in r_dex:
                scores[i][(x_de == pad_token)] = -math.inf
            attn_dist = tuple( F.softmax(scores[i],dim=1) for i in r_dex )
            context = tuple( torch.bmm(attn_dist[i].unsqueeze(1),enc_h_expand[i]).squeeze(1) for i in r_dex )
            pred = tuple( self.members[i].vocab_layer(torch.cat([dec_h[i].squeeze(1), context[i]], 1)) for i in r_dex )
            weighted_pred  = torch.stack(pred,dim=2) * out.squeeze(2)
            ensembled_pred = weighted_pred.sum(2)
            for i in r_dex:
                masterheaps[i].update_beam(ensembled_pred)
                masterheaps[i].update_hiddens(hidd[i])
                masterheaps[i].update_attentions(attn_dist[i])
                masterheaps[i].firstloop = False
        return "poop",masterheaps[0].wordlist,"morepoop"
