import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from __main__ import *
# I use EN,DE,BATCH_SIZE,MAX_LEN,pad_token,sos_token,eos_token,word2vec


# If we have beamsz 100 and we only go 3 steps, we're guaranteed to have 100 unique trigrams
# This is written so that, at any time, hiddens/attentions/wordlist/probs all align with each other across the beamsz dimension
# - We could've enforced this better by packaging together each beam member in an object, but we don't
# philosophy: we're going to take the dirty variables and reorder them nicely all in here and store them as tensors
# the inputs to these methods could have beamsz = 1 or beamsz = true beamsz (100)
class CandList():
    def __init__(self,hidden_dim,n_de,beamsz=100):
        self.beamsz = beamsz
        self.hiddens = (torch.zeros(1, 1, hidden_dim).cuda(),torch.zeros(1, 1, hidden_dim).cuda())
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
        # dimensions are nlayers*ndirections,beamsz,hiddensz
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
    def __init__(self, word_dim=300, n_layers=1, hidden_dim=500, vocab_layer_dim=500, weight_tying=False):
        super(AttnNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
         # LSTM initialization params: inputsz,hiddensz,nlayers,bias,batch_first,bidirectional
        self.encoder = nn.LSTM(word_dim, hidden_dim, num_layers = n_layers, batch_first = True)
        self.decoder = nn.LSTM(word_dim, hidden_dim, num_layers = n_layers, batch_first = True)
        self.embedding_de = nn.Embedding(len(DE.vocab), word_dim)
        self.embedding_en = nn.Embedding(len(EN.vocab), word_dim)
        if word2vec:
            self.embedding_de.weight.data.copy_(DE.vocab.vectors)
            self.embedding_en.weight.data.copy_(EN.vocab.vectors)
        # vocab layer will combine dec hidden state with context vector, and then project out into vocab space 
        # TODO: maybe put the weight tying here... Linear(hidden_dim*2,word_dim)
        self.vocab_layer = nn.Sequential(OrderedDict([
            ('h2e',nn.Linear(hidden_dim*n_layers*2, vocab_layer_dim)),
            ('tanh',nn.Tanh()),
            ('e2v',nn.Linear(vocab_layer_dim, len(EN.vocab))),
            ('lsft',nn.LogSoftmax(dim=-1))
        ]))
        if weight_tying:
            self.vocab_layer.e2v.weight.data.copy_(self.embedding_en.weight.data)
        # baseline reward, which we initialize with log 1/V
        self.baseline = Variable(torch.cuda.FloatTensor([np.log(1/len(EN.vocab))]))
        # self.baseline = Variable(torch.zeros(1).fill_(np.log(1/len(EN.vocab))).cuda()) # yoon's way
    def forward(self, x_de, x_en, attn_type="hard", update_baseline=True):
        bs = x_de.size(0)
        # x_de is bs,n_de. x_en is bs,n_en
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en) # bs,n_en,word_dim
        h0 = torch.zeros(1, bs, self.hidden_dim*self.n_layers).cuda()
        c0 = torch.zeros(1, bs, self.hidden_dim*self.n_layers).cuda()
        # hidden vars have dimension nlayers*ndirections,bs,hiddensz
        enc_h, _ = self.encoder(emb_de, (Variable(h0), Variable(c0)))
        # enc_h is bs,n_de,hiddensz*ndirections. ordering is different from last week because batch_first=True
        dec_h, _ = self.decoder(emb_en, (Variable(h0), Variable(c0)))
        # dec_h is bs,n_en,hidden_size*ndirections
        # we've gotten our encoder/decoder hidden states so we are ready to do attention
        # first let's get all our scores, which we can do easily since we are using dot-prod attention
        scores = torch.bmm(enc_h, dec_h.transpose(1,2))
        # (bs,n_de,hiddensz*ndirections) * (bs,hiddensz*ndirections,n_en) = (bs,n_de,n_en)
        reinforce_loss = 0 # we only use this variable for hard attention
        loss = 0
        avg_reward = 0
        # we just iterate to dec_h.size(1)-1, since there's </s> at the end of each sentence
        for t in range(dec_h.size(1)-1): # iterate over english words, with teacher forcing
            attn_dist = F.softmax(scores[:, :, t],dim=1) # bs,n_de. these are the alphas (attention scores for each german word)
            if attn_type == "hard":
                cat = torch.distributions.Categorical(attn_dist) 
                attn_samples = cat.sample() # bs. each element is a sample from categorical distribution
                one_hot = Variable(torch.zeros_like(attn_dist.data).scatter_(-1, attn_samples.data.unsqueeze(1), 1).cuda()) # bs,n_de
                # made a bunch of one-hot vectors
                context = torch.bmm(one_hot.unsqueeze(1), enc_h).squeeze(1)
                # now we use the one-hot vectors to select correct hidden vectors from enc_h
                # (bs,1,n_de) * (bs,n_de,hiddensz*ndirections) = (bs,1,hiddensz*ndirections). squeeze to bs,hiddensz*ndirections
            else:
                context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1) # same dimensions
                # (bs,1,n_de) * (bs,n_de,hiddensz*ndirections) = (bs,1,hiddensz*ndirections)
            # context is bs,hidden_size*ndirections
            # the rnn output and the context together make the decoder "hidden state", which is bs,2*hidden_size*ndirections
            pred = self.vocab_layer(torch.cat([dec_h[:,t,:], context], 1)) # bs,len(EN.vocab)
            y = x_en[:, t+1] # bs. these are our labels
            no_pad = (y != pad_token) # exclude english padding tokens
            reward = torch.gather(pred, 1, y.unsqueeze(1)) # bs,1
            # reward[i,1] = pred[i,y[i]]. this gets log prob of correct word for each batch. similar to -crossentropy
            reward = reward.squeeze(1)[no_pad] # less than bs
            avg_reward += reward.data.mean()
            if attn_type == "hard":
                reinforce_loss -= (cat.log_prob(attn_samples[no_pad]) * (reward-self.baseline).detach()).mean() 
                # reinforce rule (just read the formula), with special baseline
            loss -= reward.mean() # minimizing loss is maximizing reward
        avg_reward = avg_reward/dec_h.size(1)
        if update_baseline: # update baseline as a moving average
            self.baseline = Variable(0.95*self.baseline.data + 0.05*avg_reward)
        return loss, reinforce_loss
    # teacher forcing predict
    def predict(self, x_de, x_en, attn_type = "hard"):
        bs = x_de.size(0)
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en) # bs,n_en,word_dim
        h = Variable(torch.zeros(1, bs, self.hidden_dim).cuda())
        c = Variable(torch.zeros(1, bs, self.hidden_dim).cuda())
        enc_h, _ = self.encoder(emb_de, (h, c))
        dec_h, _ = self.decoder(emb_en, (h, c))
        # all the same. enc_h is bs,n_de,hiddensz*ndirections. h and c are both nlayers*ndirections,bs,hiddensz
        scores = torch.bmm(enc_h, dec_h.transpose(1,2))
        # (bs,n_de,hiddensz*ndirections) * (bs,hiddensz*ndirections,n_en) = (bs,n_de,n_en)
        y = [Variable(torch.cuda.LongTensor([sos_token]*bs))] # bs
        self.attn = []
        for t in range(x_en.size(1)-1): # iterate over english words, with teacher forcing
            attn_dist = F.softmax(scores[:,:,t],dim=1) # bs,n_de
            self.attn.append(attn_dist.data)
            if attn_type == "hard":
                _, argmax = attn_dist.max(1) # bs. for each batch, select most likely german word to pay attention to
                one_hot = Variable(torch.zeros_like(attn_dist.data).scatter_(-1, argmax.data.unsqueeze(1), 1).cuda())
                context = torch.bmm(one_hot.unsqueeze(1), enc_h).squeeze(1)
            else:
                context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1)
            # the difference btwn hard and soft is just whether we use a one_hot or a distribution
            # context is bs,hiddensz*ndirections
            pred = self.vocab_layer(torch.cat([dec_h[:,t,:], context], 1)) # bs,len(EN.vocab)
            _, next_token = pred.max(1) # bs
            y.append(next_token)
        self.attn = torch.stack(self.attn, 0).transpose(0, 1) # bs,n_en,n_de (for visualization!)
        y = torch.stack(y,0).transpose(0,1) # bs,n_en
        return y,attn
    # Singleton batch with BSO
    def predict2(self, x_de, beamsz, gen_len, attn_type = "hard"):
        emb_de = self.embedding_de(x_de) # "batch size",n_de,word_dim, but "batch size" is 1 in this case!
        h0 = Variable(torch.zeros(1, 1, self.hidden_dim).cuda())
        c0 = Variable(torch.zeros(1, 1, self.hidden_dim).cuda())
        enc_h, _ = self.encoder(emb_de, (h0, c0))
        # since enc batch size=1, enc_h is 1,n_de,hiddensz*ndirections
        masterheap = CandList(self.hidden_dim,enc_h.size(1),beamsz)
        # in the following loop, beamsz is length 1 for first iteration, length true beamsz (100) afterward
        for i in range(gen_len):
            prev = masterheap.get_prev() # beamsz
            emb_t = self.embedding_en(prev) # embed the last thing we generated. beamsz,word_dim
            enc_h_expand = enc_h.expand(prev.size(0),-1,-1) # beamsz,n_de,hiddensz*ndirections
            h, c = masterheap.get_hiddens() # (nlayers*ndirections,beamsz,hiddensz),(nlayers*ndirections,beamsz,hiddensz)
            dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c)) # dec_h is beamsz,1,hiddensz*ndirections (batch_first=True)
            scores = torch.bmm(enc_h_expand, dec_h.transpose(1,2)).squeeze(2)
            # (beamsz,n_de,hiddensz*ndirections) * (beamsz,hiddensz*ndirections,1) = (beamsz,n_de,1). squeeze to beamsz,n_de
            attn_dist = F.softmax(scores,dim=1)
            if attn_type == "hard":
                _, argmax = attn_dist.max(1) # beamsz for each batch, select most likely german word to pay attention to
                one_hot = Variable(torch.zeros_like(attn_dist.data).scatter_(-1, argmax.data.unsqueeze(1), 1).cuda())
                context = torch.bmm(one_hot.unsqueeze(1), enc_h_expand).squeeze(1)
            else:
                context = torch.bmm(attn_dist.unsqueeze(1), enc_h_expand).squeeze(1)
            # the difference btwn hard and soft is just whether we use a one_hot or a distribution
            # context is beamsz,hiddensz*ndirections
            pred = self.vocab_layer(torch.cat([dec_h.squeeze(1), context], 1)) # beamsz,len(EN.vocab)
            # TODO: set the columns corresponding to <pad>,<unk>,</s>,etc to 0
            masterheap.update_beam(pred)
            masterheap.update_hiddens(h,c)
            masterheap.update_attentions(attn_dist)
            masterheap.firstloop = False
        return masterheap.probs,masterheap.wordlist,masterheap.attentions


class S2S(nn.Module):
    def __init__(self, word_dim=300, hidden_dim=500):
        super(S2S, self).__init__()
        self.hidden_dim = hidden_dim
        # LSTM initialization params: inputsz,hiddensz,nlayers,bias,batch_first,bidirectional
        self.encoder = nn.LSTM(word_dim, hidden_dim, num_layers = 1, batch_first = True)
        self.decoder = nn.LSTM(word_dim, hidden_dim, num_layers = 1, batch_first = True)
        self.embedding_de = nn.Embedding(len(DE.vocab), word_dim)
        self.embedding_en = nn.Embedding(len(EN.vocab), word_dim)
        if word2vec:
            self.embedding_de.weight.data.copy_(DE.vocab.vectors)
            self.embedding_en.weight.data.copy_(EN.vocab.vectors)
        # vocab layer will project dec hidden state out into vocab space 
        self.vocab_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         nn.Tanh(), nn.Linear(hidden_dim, len(EN.vocab)), nn.LogSoftmax(dim=-1))
        self.baseline = Variable(torch.zeros(1)) # just doing this (and attn_type below) so I can reuse some code lol
    def forward(self, x_de, x_en, attn_type):
        bs = x_de.size(0)
        # x_de is bs,n_de. x_en is bs,n_en
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en) # bs,n_en,word_dim
        h = Variable(torch.zeros(1, bs, self.hidden_dim).cuda())
        c = Variable(torch.zeros(1, bs, self.hidden_dim).cuda())
        # hidden vars have dimension nlayers*ndirections,bs,hiddensz
        enc_h, (h,c) = self.encoder(emb_de, (h, c))
        # enc_h is bs,n_de,hiddensz*ndirections. ordering is different from last week because batch_first=True
        dec_h, _ = self.decoder(emb_en, (h, c))
        # dec_h is bs,n_en,hidden_size*ndirections
        pred = self.vocab_layer(dec_h) # bs,n_en,len(EN.vocab)
        pred = pred[:,:-1,:] # alignment
        y = x_en[:,1:]
        no_pad = (y != pad_token)
        reward = torch.gather(pred,2,y.unsqueeze(2))
        # reward[i,j,1] = pred[i,j,y[i,j]]
        reward = reward.squeeze(2)[no_pad] # less than bs,n_en
        loss = -reward.sum() / no_pad.data.sum() * bs
        # NOTE: to be consistent with the other network i'm normalizing loss by time only (not by batch) 
        return loss, 0 # passing back an invisible "negative reward"
    # predict with greedy decoding
    def predict(self, x_de, attn_type):
        bs = x_de.size(0)
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        h = Variable(torch.zeros(1, bs, self.hidden_dim).cuda())
        c = Variable(torch.zeros(1, bs, self.hidden_dim).cuda())
        enc_h, (h,c) = self.encoder(emb_de, (h, c))
        # all the same. enc_h is bs,n_de,hiddensz*ndirections. h and c are both nlayers*ndirections,bs,hiddensz
        y = [Variable(torch.cuda.LongTensor([sos_token]*bs))] # bs
        n_en = MAX_LEN # this will change
        for t in range(n_en): # generate some english.
            emb_t = self.embedding_en(y[-1]) # embed the last thing we generated. bs
            dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c)) # dec_h is bs,1,hiddensz*ndirections (batch_first=True)
            pred = self.vocab_layer(dec_h) # bs,1,len(EN.vocab)
            _, next_token = pred.max(1) # bs
            y.append(next_token)
        return torch.stack(y, 0).transpose(0, 1) # bs,n_en
