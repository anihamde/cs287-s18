import numpy as np
import torch
import torch.nn as nn

# I thought it might be better to move these unwieldy models into their own file. Feel free to change it back!

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

    # alternate function: singleton batch. store stuff in a heap, somehow
    
    # alternate function: singleton batch. store stuff in a heap, somehow
    def predict2(self, x_de, beamsz, gen_len=3, attn_type = "hard"):
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        h0 = Variable(torch.zeros(1, 1, self.hidden_dim))
        c0 = Variable(torch.zeros(1, 1, self.hidden_dim))
        enc_h, _ = self.encoder(emb_de, (h0, c0))
        # bs is 1 in this case. singleton batch!
        # hence, enc_h is 1,n_de,hiddensz*ndirections. h and c are both nlayers*ndirections,1,hiddensz
        masterheap = CandList(beamsz,self.hidden_dim,enc_h.size(1))
        beam = None
        for i in range(gen_len):
            candlist = masterheap.get_candlist() # beamsz
            beamsz = candlist.size(0)
            h, c = masterheap.get_hiddens() # (nlayers*ndirections,beamsz,hiddensz),(nlayers*ndirections,beamsz,hiddensz)
            attn = masterheap.get_attentions() # beamsz,i,n_de
            emb_t = self.embedding(candlist) # embed the last thing we generated. beamsz,word_dim
            dec_h, (h, c) = self.decoder(candlist.unsqueeze(1), (h, c)) # dec_h is beamsz,1,hiddensz*ndirections (batch_first=True)
            scores = torch.bmm(enc_h.expand(beamsz,-1,-1), dec_h.transpose(1,2)).squeeze(2)
            # (beamsz,n_de,hiddensz*ndirections) * (beamsz,hiddensz*ndirections,1) = (beamsz,n_de,1). squeeze to beamsz,n_de
            attn_dist = F.softmax(scores,dim=1)
            attn_dist.data # will be important to save these TODO!
            if attn_type == "hard":
                _, argmax = attn_dist.max(1) # bs. for each batch, select most likely german word to pay attention to
                one_hot = Variable(torch.zeros_like(attn_dist.data).scatter_(-1, argmax.data.unsqueeze(1), 1))
                context = torch.bmm(one_hot.unsqueeze(1), enc_h.expand(beamsz,-1,-1)).squeeze(1)
            else:
                context = torch.bmm(attn_dist.unsqueeze(1), enc_h.expand(beamsz,-1,-1)).squeeze(1)
            # the difference btwn hard and soft is just whether we use a one_hot or a distribution
            # context is beamsz,hiddensz*ndirections
            pred = self.vocab_layer(torch.cat([dec_h.squeeze(1), context], 1)) # beamsz,len(EN.vocab)
            newlogprobs = pred
            beam = masterheap.update_candlist(beam,newlogprobs) # (beamsz)x(iter+1)
            masterheap.update_hiddens(h,c)
            masterheap.update_attns(attn_dist)
        
        return beam
            
# TODO: not holding onto variables forever?
        
# If we have beamsz 100 and we only go 3 steps, we're guaranteed to have 100 unique trigrams
class CandList():
    def __init__(self,beamsz=100,hidden_dim,n_de):
        self.candlist = [[sos.token]]
        self.hiddens = (torch.zeros(1, beamsz, hidden_dim),torch.zeros(1, beamsz, hidden_dim))
        self.attentions = torch.zeros(beamsz,1,n_de)
    def get_candlist():
        return Variable(torch.LongTensor(self.candlist))
    def get_hiddens():
        return Variable(self.hiddens[0],self.hiddens[1])
    def get_attentions():
        return self.attentions
    def update_candlist(wordslist,newlogprobs):
        if wordslist:
            newlogprobs += self.probs
            newlogprobs = torch.flatten(newlogprobs)
            sorted, indices = torch.sort(newlogprobs,[beamsz])
            self.probs = sorted
            self.oldbeamindices = indices/len(EN.vocab)            
            currbeam = indices%len(EN.vocab).unsqueeze(1) # (beamsz)x(1)
            
            wlizt = []
            
            prevbeam = wordslist[self.oldbeamindices] # (beamsz)x(iter+1)
            
            fullbeam = torch.cat([prevbeam,currbeam],1) # (beamsz)x(iter+1)
        else:
            sorted, indices = torch.sort(newlogprobs,[beamsz])
            self.probs = sorted
            self.oldbeamindices = None

            fullbeam = indices.unsqueeze(1) # (beamsz)x(1)
            
        return fullbeam
        
      def update_hiddens(self,h,c):
          h_new = h[:,self.oldbeamindices,:].data
          c_new = c[:,self.oldbeamindices,:].data
          self.hiddens = (h_new,c_new)
          
      def update_attentions(self,attn):
          shuffled = self.attentions[self.oldbeamindices,:,:]
          self.attentions = torch.cat([shuffle

''' pseudocode
y = [bs]
heap = {}
insert y into the heap, with its attention matrix and logprob 0
finished_sentences = {}
while finished_sentences isn't full and heap isn't empty
    pred = run_the_model_forward
    toplogprobs,topindices=torch.max(preds, only the top 20 tho)
    
'''

class S2S(nn.Module):
    def __init__(self, word_dim=300, hidden_dim=500):
        super(S2S, self).__init__()
        self.hidden_dim = hidden_dim
        # LSTM initialization params: inputsz,hiddensz,nlayers,bias,batch_first,bidirectional
        self.encoder = nn.LSTM(word_dim, hidden_dim, num_layers = 1, batch_first = True)
        self.decoder = nn.LSTM(word_dim, hidden_dim, num_layers = 1, batch_first = True)
        self.embedding_de = nn.Embedding(len(DE.vocab), word_dim)
        self.embedding_en = nn.Embedding(len(EN.vocab), word_dim)
        # vocab layer will project dec hidden state out into vocab space 
        self.vocab_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         nn.Tanh(), nn.Linear(hidden_dim, vocab_size), nn.LogSoftmax())               
        
    def forward(self, x_de, x_en, bs=BATCH_SIZE):
        # x_de is bs,n_de. x_en is bs,n_en
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        emb_en = self.embedding_en(x_en) # bs,n_en,word_dim
        h = Variable(torch.zeros(1, bs, self.hidden_dim)) 
        c = Variable(torch.zeros(1, bs, self.hidden_dim))
        # hidden vars have dimension nlayers*ndirections,bs,hiddensz
        enc_h, (h,c) = self.encoder(emb_de, (h, c))
        # enc_h is bs,n_de,hiddensz*ndirections. ordering is different from last week because batch_first=True
        dec_h, _ = self.decoder(emb_en, (h, c))
        # dec_h is bs,n_en,hidden_size*ndirections
        pred = self.vocab_layer(dec_h) # bs,n_en,len(EN.vocab)
        pred = pred[:,:-1]
        y = x_en[:,1:] # bs,n_en
        reward = torch.gather(pred,2,y.unsqueeze(2))
        # reward[i][j][k] = pred[i][j][y[i][j][k]]
        loss = -reward.sum()/bs
        # TODO: to be consistent with the other network i'm not dividing by n_en here. Can we change this?
        return loss, 0 # passing back an invisible "negative reward"
    
    def predict(self, x_de, bs=BATCH_SIZE):
        # predict with greedy decoding
        emb_de = self.embedding_de(x_de) # bs,n_de,word_dim
        h = Variable(torch.zeros(1, bs, self.hidden_dim))
        c = Variable(torch.zeros(1, bs, self.hidden_dim))
        enc_h, (h,c) = self.encoder(emb_de, (h, c))
        # all the same. enc_h is bs,n_de,hiddensz*ndirections. h and c are both nlayers*ndirections,bs,hiddensz
        y = [Variable(torch.LongTensor([sos.token]*bs))] # bs
        n_en = MAX_LEN # this will change
        for t in range(n_en): # generate some english.
            emb_t = self.embedding(y[-1]) # embed the last thing we generated. bs
            dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c)) # dec_h is bs,1,hiddensz*ndirections (batch_first=True)
            pred = self.vocab_layer(dec_h) # bs,1,len(EN.vocab)
            _, next_token = pred.max(1) # bs
            y.append(next_token)
        return torch.stack(y, 0).transpose(0, 1) # bs,n_en