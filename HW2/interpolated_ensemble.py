import numpy as np
from collections import Counter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv
import argparse
import sys

parser = argparse.ArgumentParser(description='lstm training runner')
parser.add_argument('--model_file','-m',type=str,default='../../models/HW2/interpolated_ensemble.pkl',help='Model save target.')
parser.add_argument('--batch_size','-bs',type=int,default=10,help='set training batch size. default=10.')
parser.add_argument('--num_layers','-nl',type=int,default=2,help='set number of lstm layers.')
parser.add_argument('--hidden_size','-hs',type=int,default=500,help='set size of hidden layer.')
parser.add_argument('--receptive_field','-rf',type=int,default=5,help='set receptive field of nnlm.')
parser.add_argument('--convolutional_featuremap_1','-cf1',type=int,default=200,help='Featuremap density for 3x1 conv.')
parser.add_argument('--convolutional_featuremap_2','-cf2',type=int,default=200,help='Featuremap density for 5x1 conv.')
parser.add_argument('--learning_rate','-lr',type=float,default=0.001,help='set learning rate.')
parser.add_argument('--weight_decay','-wd',type=float,default=0.0,help='set L2 normalization factor.')
parser.add_argument('--num_epochs','-e',type=int,default=5,help='set the number of training epochs.')
parser.add_argument('--embedding_max_norm','-emn',type=float,default=15,help='set max L2 norm of word embedding vector.')
parser.add_argument('--dropout_rate','-dr',type=float,default=0.5,help='set dropout rate for deep layers.')
parser.add_argument('--skip_training','-sk',action='store_true',help='raise flag to skip training and go to eval.')
parser.add_argument('--clip_constraint','-c',type=float,default=5,help='set constraint for gradient clipping.')
args = parser.parse_args()

# Hyperparameters
bs = args.batch_size
n = args.receptive_field # receptive field
n_layers = args.num_layers
hidden_size = args.hidden_size
learning_rate = args.learning_rate
weight_decay = args.weight_decay
num_epochs = args.num_epochs
emb_mn = args.embedding_max_norm
dropout_rate = args.dropout_rate
constraint = args.clip_constraint
n_featmaps1 = args.convolutional_featuremap_1
n_featmaps2 = args.convolutional_featuremap_2

# Text processing library
import torchtext
from torchtext.vocab import Vectors
# Our input $x$
TEXT = torchtext.data.Field()
# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="./", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

print('len(train)', len(train), file=sys.stderr)

TEXT.build_vocab(train)
padidx = TEXT.vocab.stoi["<pad>"]
print('len(TEXT.vocab)', len(TEXT.vocab), file=sys.stderr)

if False:
    TEXT.build_vocab(train, max_size=1000)
    len(TEXT.vocab)

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=bs, device=-1, bptt_len=32, repeat=False)

it = iter(train_iter)
batch = next(it) 
print("Size of text batch [max bptt length, batch size]", batch.text.size(), file=sys.stderr)
print("Second in batch", batch.text[:, 2], file=sys.stderr)
print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 2].data]), file=sys.stderr)
batch = next(it)
print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 2].data]), file=sys.stderr)

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) # feel free to alter path
print("Word embeddings size ", TEXT.vocab.vectors.size(), file=sys.stderr)
word2vec = TEXT.vocab.vectors
print("REMINDER!!! Did you create ../../models/HW2?????", file=sys.stderr)

# TODO: attention!!
# TODO: learning rate decay? (zaremba has specific instructions for this)
# TODO: bidirectional, gru

# TODO: minibatch size 20 (according to zaremba), and clip the grads normalized by minibatch size
# TODO: clip grads doesn't work- it deletes all the parameters, and returns norm of 0?
# TODO: multichannel tests (with glove and stuff), more of our own ideas

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

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
        out = self.u(F.dropout(out,p=dropout_rate)) # bs,|V|
        embeds = F.dropout(embeds,p=dropout_rate)
        out += self.w(embeds) # bs,|V|
        #out = F.softmax(out,dim=1)
        return out
        
class dLSTM(nn.Module):
    def __init__(self):
        super(dLSTM, self).__init__()
        self.embedding = nn.Embedding(word2vec.size(0),word2vec.size(1),max_norm=emb_mn)
        self.embedding.weight.data.copy_(word2vec)
        self.lstm = nn.LSTM(word2vec.size(1), hidden_size, n_layers, dropout=dropout_rate)
        self.linear = nn.Linear(hidden_size, len(TEXT.vocab))
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, input, hidden): 
        # input is (sentence length, batch size) n,bs
        # hidden is ((n_layers,bs,hidden_size),(n_layers,bs,hidden_size))
        embeds = self.embedding(input) # n,bs,300
        # batch goes along the second dimension
        out = F.dropout(embeds,p=dropout_rate)
        out, hidden = self.lstm(out, hidden)
        out = F.dropout(out,p=dropout_rate)
        # apply the linear and the softmax
        out = self.linear(out) # n,bs,|V|
        #out = self.softmax(out)    # This was originally the output. (SG: I see this is LogSoftmax)
        return out, hidden
    
    def initHidden(self):
        h0 = torch.zeros(n_layers, bs, hidden_size).type(torch.FloatTensor)
        c0 = torch.zeros(n_layers, bs, hidden_size).type(torch.FloatTensor)
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (Variable(h0), Variable(c0))
    
class dGRU(nn.Module):
    def __init__(self):
        super(dGRU, self).__init__()
        self.embedding = nn.Embedding(word2vec.size(0),word2vec.size(1),max_norm=emb_mn)
        self.embedding.weight.data.copy_(word2vec)
        self.gru = nn.GRU(word2vec.size(1), hidden_size, n_layers, dropout=dropout_rate)
        self.linear = nn.Linear(hidden_size, len(TEXT.vocab))
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, input, hidden): 
        # input is (sentence length, batch size) n,bs
        # hidden is ((n_layers,bs,hidden_size),(n_layers,bs,hidden_size))
        embeds = self.embedding(input) # n,bs,300
        # batch goes along the second dimension
        out = F.dropout(embeds,p=dropout_rate)
        out, hidden = self.gru(out, hidden)
        out = F.dropout(out,p=dropout_rate)
        # apply the linear and the softmax
        out = self.linear(out) # n,bs,|V|
        #out = self.softmax(out)    # This was originally the output. (SG: I see this is LogSoftmax)
        return out, hidden
    
    def initHidden(self):
        h0 = torch.zeros(n_layers, bs, hidden_size).type(torch.FloatTensor)
        if torch.cuda.is_available():
            h0 = h0.cuda()
        return Variable(h0)

class Alpha(nn.Module):
    def __init__(self):
        super(Alpha, self).__init__()
        self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
        self.embeddings.weight.data = TEXT.vocab.vectors
        self.conv3 = nn.Conv2d(300,n_featmaps1,kernel_size=(3,1),padding=(1,0))
        self.conv5 = nn.Conv2d(300,n_featmaps2,kernel_size=(5,1),padding=(2,0))
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(n_featmaps1+n_featmaps2,3)
        
    def forward(self, inputs, model1, model2, model3):
        bsz = inputs.size(0) # batch size might change
        if inputs.size(1) < 3: # padding issues on really short sentences
            pads = Variable(torch.zeros(bsz,3-inputs.size(1))).type(torch.LongTensor)
            inputs = torch.cat([inputs,pads.cuda()],dim=1)
        embeds = self.embeddings(inputs) # bs,h,300
        out = embeds.unsqueeze(1) # bs,1,h,300
        out = out.permute(0,3,2,1) # bs,300,h,1
        fw3 = self.conv3(out) # bs,n_feat1,h,1
        fw5 = self.conv5(out) # bs,n_feat2,h,1
        out = torch.cat([fw3,fw5],dim=1)
        out = F.relu(out) # bs,n_feat1+n_feat2,h,1
        out = out.squeeze(-1) # bs,n_feat1+n_feat2,h
        out = self.maxpool(out) # bs,n_feat1+n_feat2,1
        out = out.squeeze(-1) # bs,n_feat1+n_feat2
        out = self.dropout(out) # bs,n_feat1+n_feat2
        out = self.linear(out) # bs,3
        out = F.softmax(out,dim=1)
        model_stack = torch.stack([model1,model2,model3],dim=3)
        out = out.unsqueeze(0)
        out = out.unsqueeze(2)
        print(out.size())
        print(model_stack.size())
        out = model_stack * out
        out = out.sum(3)
        return out
    
fNNLM = NNLM()    
fLSTM = dLSTM()    
fGRU = dGRU()
fNNLM.load_state_dict(torch.load('../../models/HW2/nnlm.pkl'))
fLSTM.load_state_dict(torch.load('../../models/HW2/lstm.pkl'))
fGRU.load_state_dict(torch.load('../../models/HW2/gru.pkl'))
freeze_model(fNNLM)
freeze_model(fLSTM)
freeze_model(fGRU)
fNNLM.cuda()
fLSTM.cuda()
fGRU.cuda()
fNNLM.eval()
fLSTM.eval()
fGRU.eval()

model = Alpha()
if torch.cuda.is_available():
    model.cuda()
    print("CUDA is available, assigning to GPU.", file=sys.stderr)

criterion = nn.CrossEntropyLoss()
params = filter(lambda x: x.requires_grad, model.parameters())
biasparams = []
for name,param in model.named_parameters():
    if 'bias' in name:
        biasparams.append(param)

optimizer = torch.optim.Adam(biasparams, lr=learning_rate, weight_decay=weight_decay)

def validate():
    softmaxer = torch.nn.Softmax(dim=1)
    model.eval()
    correct = total = 0
    precisionmat = (1/np.arange(1,21))[::-1].cumsum()[::-1]
    precisionmat = torch.cuda.FloatTensor(precisionmat.copy())
    precision = 0
    crossentropy = 0
    LSTMhidden = fLSTM.initHidden()
    GRUhidden = fGRU.initHidden()
    for batch in iter(val_iter):
        sentences = batch.text # n=32,bs
        if torch.cuda.is_available():
            sentences = sentences.cuda()
        LSTMout, LSTMhidden = fLSTM(sentences, LSTMhidden)
        GRUout, GRUhidden = fGRU(sentences, GRUhidden)
        word_pad = (32 + n - 1) - sentences.size(0)
        pads = Variable(torch.zeros(word_pad,sentences.size(1))).type(torch.cuda.LongTensor)
        padsentences = torch.cat([pads,sentences],dim=0)
        #print("sentence_dim: {}\npadded_dim: {}".format(sentences.size(),padsentences.size()))
        NNLMout = torch.stack([ fNNLM(torch.cat([ padsentences[:,a:a+1][b:b+n,:] for b in range(32) ],dim=1).t()) for a in range(sentences.size(1)) ],dim=1)
        #eOUT = torch.cat([LSTMout,GRUout,NNLMout],dim=2)
        NNLMout = NNLMout[-sentences.size(0):,:sentences.size(1),:len(TEXT.vocab)]
        tOUT = model(sentences.t(),LSTMout,GRUout,NNLMout)
        out  = tOUT
        for j in range(sentences.size(0)-1):
            outj = out[j] # bs,|V|
            labelsj = sentences[j+1] # bs
            # cross entropy
            crossentropy += F.cross_entropy(outj,labelsj,size_average=False,ignore_index=padidx)
            # precision
            outj, labelsj = softmaxer(outj).data, labelsj.data
            _, outsort = torch.sort(outj,dim=1,descending=True)
            outsort = outsort[:,:20]
            inds = (outsort-labelsj.unsqueeze(1)==0)
            inds = inds.sum(dim=0).type(torch.cuda.FloatTensor)
            precision += inds.dot(precisionmat)
            # plain ol accuracy
            _, predicted = torch.max(outj, 1)
            total += labelsj.ne(padidx).int().sum()
            correct += (predicted==labelsj).sum()
            # DEBUGGING: see the rest in trigram.py
        LSTMhidden = repackage_hidden(LSTMhidden)
        GRUhidden  = repackage_hidden(GRUhidden)
    return correct/total, precision/total, torch.exp(crossentropy/total).data[0]
        # test acc, precision, ppl
        # F.cross_entropy averages instead of adding


if not args.skip_training:
    losses = []
    for i in range(num_epochs):
        model.train()
        ctr = 0
        # initialize hidden vector
        LSTMhidden = fLSTM.initHidden()
        GRUhidden = fGRU.initHidden()
        for batch in iter(train_iter):
            sentences = batch.text # Variable of LongTensor of size (n,bs)
            if torch.cuda.is_available():
                sentences = sentences.cuda()
            LSTMout, LSTMhidden = fLSTM(sentences, LSTMhidden)
            GRUout, GRUhidden = fGRU(sentences, GRUhidden)
            word_pad = (32 + n - 1) - sentences.size(0)
            pads = Variable(torch.zeros(word_pad,sentences.size(1))).type(torch.cuda.LongTensor)
            padsentences = torch.cat([pads,sentences],dim=0)
            #print("sentence_dim: {}\npadded_dim: {}".format(sentences.size(),padsentences.size()))
            NNLMout = torch.stack([ fNNLM(torch.cat([ padsentences[:,a:a+1][b:b+n,:] for b in range(32) ],dim=1).t()) for a in range(sentences.size(1)) ],dim=1)
            NNLMout = NNLMout[-sentences.size(0):,:sentences.size(1),:len(TEXT.vocab)]
            #print("nnlm_dim: {}".format(NNLMout.size()))

            #print("gru_dim: {}".format(GRUout.size()))
            # out is n,bs,|V|, hidden is ((n_layers,bs,hidden_size)*2)
            #eOUT = torch.cat([LSTMout,GRUout,NNLMout],dim=2)
            tOUT = model(sentences.t(),LSTMout,GRUout,NNLMout)
            out  = tOUT
            loss = criterion(out[:-1,:,:].view(-1,10001), sentences[1:,:].view(-1))
            model.zero_grad()
            loss.backward(retain_graph=True)
            #nn.utils.clip_grad_norm(params, constraint, norm_type=2) # what the, why is it zero
            optimizer.step()
            # hidden vector is automatically saved for next batch
            ctr += 1
            losses.append(loss.data[0])
            if ctr % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                    %(i+1, num_epochs, ctr, len(train_iter), sum(losses[-500:])/len(losses[-500:])  ),
                      file=sys.stderr)
            LSTMhidden = repackage_hidden(LSTMhidden)
            GRUhidden = repackage_hidden(GRUhidden)

        # can add a net_flag to these file names. and feel free to change the paths
        np.save("../../models/HW2/interpolated_ensemble_losses",np.array(losses))
        torch.save(model.state_dict(), args.model_file)
        # for early stopping
        acc, prec, ppl = validate()
        print("Val acc, prec, ppl", acc, prec, ppl)
else:
    model.load_state_dict(torch.load(args.model_file))


model.eval()
model.eval()
with open("interpolated_ensemble_predictions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['id','word'])
    for i, l in enumerate(open("input.txt"),1):
        words = [TEXT.vocab.stoi[word] for word in l.split(' ')]
        words = Variable(torch.cuda.LongTensor(words).unsqueeze(1))
        LSTMhidden = (Variable(torch.zeros(n_layers, 1, hidden_size)).cuda(),
            Variable(torch.zeros(n_layers, 1, hidden_size)).cuda())
        GRUhidden = Variable(torch.zeros(n_layers, 1, hidden_size)).cuda()
        LSTMout, LSTMhidden = fLSTM(words, LSTMhidden)
        GRUout, GRUhidden = fGRU(words, GRUhidden)
        pads = Variable(torch.zeros(n-1,words.size(1))).type(torch.cuda.LongTensor)
        padwords = torch.cat([pads,words],dim=0)
        NNLMout = fNNLM(torch.cat([ padwords[:,0:1][b:b+n,:] for b in range(words.size(0)) ],dim=1).t()).unsqueeze(1)
        #print(NNLMout.size())
        out = model(words.t(),LSTMout,GRUout,NNLMout)
        out = out.view(-1,len(TEXT.vocab))[-2]
        #out = out.squeeze(1)[-2] # |V|
        out = F.softmax(out,dim=0)
        _, predicted = torch.sort(out,descending=True)
        predicted = predicted[:20].data.tolist()
        predwords = [TEXT.vocab.itos[x] for x in predicted]
        writer.writerow([i,' '.join(predwords)])
