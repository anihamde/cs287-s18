import argparse

parser = argparse.ArgumentParser(description='To be (attentive) or not to be (attentive)? That is the question.')
parser.add_argument('attention', metavar='is_attent', type=str, nargs=1,
                    help='Whether or not to have attention')

args = parser.parse_args()
is_attent = args[0]

eta = 0.05

class AttnNetwork(nn.Module):
    def __init__(self, vocab_size_DE = 50, vocab_size_EN = 50, word_dim = 50, hidden_dim = 300):
        super(AttnNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(word_dim, hidden_dim, num_layers = 1, batch_first = True)
        self.decoder = nn.LSTM(word_dim, hidden_dim, num_layers = 1, batch_first = True)
        self.embedding_DE = nn.Embedding(vocab_size_DE, word_dim) #we are going to be sharing the embedding layer 
        #this vocab layer will combine dec hidden state with context vector, and the project out into vocab space 
        self.embedding_EN = nn.Embedding(vocab_size_EN, word_dim)
        if is_attent:
            self.vocab_layer = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                         nn.Tanh(), nn.Linear(hidden_dim, vocab_size_EN), nn.LogSoftmax())
            #baseline reward, which we initialize with log 1/V
        else:
            self.vocab_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         nn.Tanh(), nn.Linear(hidden_dim, vocab_size_EN), nn.LogSoftmax())
            #baseline reward, which we initialize with log 1/V
        self.baseline = Variable(torch.zeros(1).fill_(np.log(1/vocab_size_EN)))                
        
    def forward(self, x_DE, x_EN, attn_type="hard"):
        emb_DE = self.embedding_DE(x_DE)
        emb_EN = self.embedding_EN(x_EN)
        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim))
        c0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim))
        enc_h, _ = self.encoder(emb_DE, (h0, c0))
        dec_h, _ = self.decoder(emb_EN[:, :-1], (h0, c0))
        #we've gotten our encoder/decoder hidden states so we are ready to do attention        
        #first let's get all our scores, which we can do easily since we are using dot-prod attention
        scores = torch.bmm(enc_h, dec_h.transpose(1,2)) #this will be a batch x source_len x target_len
        neg_reward = 0
        loss = 0
        avg_reward = 0        
        if is_attent == "YES":
            for t in range(dec_h.size(1)):            
                attn_dist = F.softmax(scores[:, :, t], dim=1) #get attention scores
                if attn_type == "hard":
                    cat = torch.distributions.Categorical(attn_dist) 
                    attn_samples = cat.sample() #samples from attn_dist    
                    #make this into a one-hot distribution (there are more efficient ways of doing this)
                    one_hot = Variable(torch.zeros_like(attn_dist.data).scatter_(-1, attn_samples.data.unsqueeze(1), 1))
                    context = torch.bmm(one_hot.unsqueeze(1), enc_h).squeeze(1)                 
                else:
                    context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1)
                pred = self.vocab_layer(torch.cat([dec_h[:, t], context], 1))
                y = x[:, t+1] #this will be our label
                reward = torch.gather(pred, 1, y.unsqueeze(1))  #our reward is log prob at the word level
                avg_reward += reward.data.mean() 
                if attn_type == "hard":                
                    neg_reward -= (cat.log_prob(attn_samples) * (reward.detach()-self.baseline)).mean() #reinforce rule                                        
                loss -= reward.mean()
        else:
            for t in range(dec_h.size(1)):            
                pred = self.vocab_layer(dec_h[:, t])
                y = x[:, t+1] #this will be our label
                reward = torch.gather(pred, 1, y.unsqueeze(1))  #our reward is log prob at the word level
                avg_reward += reward.data.mean()                                     
                loss -= reward.mean()
        avg_reward = avg_reward/dec_h.size(1)
        self.baseline.data = 0.95*self.baseline.data + 0.05*avg_reward #update baseline as a moving average
        return loss, neg_reward
    
    def predict(self, x, attn_type = "hard"):
        #predict with greedy decoding
        emb = self.embedding_DE(x)
        h = Variable(torch.zeros(1, x.size(0), self.hidden_dim))
        c = Variable(torch.zeros(1, x.size(0), self.hidden_dim))
        enc_h, _ = self.encoder(emb, (h, c))
        y = [Variable(torch.zeros(x.size(0)).long())]
        if is_attent:
            self.attn = []        
            for t in range(x.size(1)):
                emb_t = self.embedding_DE(y[-1])
                dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c))
                scores = torch.bmm(enc_h, dec_h.transpose(1,2)).squeeze(2)
                attn_dist = F.softmax(scores, dim = 1)
                self.attn.append(attn_dist.data)
                if attn_type == "hard":
                    _, argmax = attn_dist.max(1)
                    one_hot = Variable(torch.zeros_like(attn_dist.data).scatter_(-1, argmax.data.unsqueeze(1), 1))
                    context = torch.bmm(one_hot.unsqueeze(1), enc_h).squeeze(1)                    
                else:                
                    context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1)
                pred = self.vocab_layer(torch.cat([dec_h.squeeze(1), context], 1))
                _, next_token = pred.max(1)
                y.append(next_token)
            self.attn = torch.stack(self.attn, 0).transpose(0, 1)
        else:
            for t in range(x.size(1)):
                emb_t = self.embedding_DE(y[-1])
                dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c))
                scores = torch.bmm(enc_h, dec_h.transpose(1,2)).squeeze(2)
                pred = self.vocab_layer(dec_h.squeeze(1))
                _, next_token = pred.max(1)
                y.append(next_token)
        return torch.stack(y, 0).transpose(0, 1)


def pretty_print(t):
    #print 2d tensors nicely
    print("\n".join([" ".join(list(map("{:.3f}".format, list(u)))) for u in list(t)]))
    print('')
    
model = AttnNetwork()
attn_type = "hard"
num_iters = 10000
optim = torch.optim.SGD(model.parameters(), lr=eta)
avg_acc = 0
for i in range(num_iters):
    optim.zero_grad()
    x, y = generate_data()
    loss, neg_reward = model.forward(x, attn_type)
    y_pred = model.predict(x, attn_type)
    correct = torch.sum(y_pred.data[:, 1:] == y.data[:, 1:]) #exclude <s> token in acc calculation    
    (loss + neg_reward).backward()
    # torch.nn.utils.clip_grad_norm(model.parameters(), 1)
    optim.step()     
    avg_acc = 0.95*avg_acc + 0.05*correct/ (x.size(0)*x.size(1))    
    if i % 100 == 0:        
        print("Attn Type: %s, Iter: %d, Reward: %.2f, Accuracy: %.2f, PPL: %.2f" %
              (attn_type, i, model.baseline.data[0], avg_acc, np.exp(loss/x.size(1))))
        pretty_print(model.attn[0])        
            