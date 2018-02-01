# Text processing library and methods for pretrained word embeddings
import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchtext.vocab import Vectors, GloVe
import csv
import sys

# Hyperparams
filter_window = 3
n_featmaps = 100
n_featmaps2= 200
bs = 10
dropout_rate = 0.5
num_epochs = 15 # from 30
learning_rate = 0.001
constraint = 3

# Our input $x$
TEXT = torchtext.data.Field()
# Our labels $y$
LABEL = torchtext.data.Field(sequential=False)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

TEXT.build_vocab(train)
LABEL.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))
print('len(LABEL.vocab)', len(LABEL.vocab))

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=bs, device=-1, repeat=False)

# glove = GloVe(name='6B',dim=300)
TEXT.vocab.load_vectors(vectors=Vectors('glove.6B.300d.txt'))
glove = TEXT.vocab.vectors

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

print("Word embeddings size ", TEXT.vocab.vectors.size())
word2vec = TEXT.vocab.vectors

############################################
# With help from Yunjey Pytorch Tutorial on Github

if sys.argv[1] == None:
    net_flag = 'normal'
else:
    net_flag = sys.argv[1]

assert net_flag in ['normal','multi','extralayer','dialated','']

if net_flag == 'normal':
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
            self.embeddings.weight.data = TEXT.vocab.vectors
            self.conv = nn.Conv2d(n_featmaps*3,n_featmaps2,kernel_size=(filter_window,1),padding=(1,0))
            self.conv3 = nn.Conv2d(300,n_featmaps,kernel_size=(3,1),padding=(1,0))
            self.conv5 = nn.Conv2d(300,n_featmaps,kernel_size=(5,1),padding=(2,0))
            self.conv7 = nn.Conv2d(300,n_featmaps,kernel_size=(7,1),padding=(3,0))
            self.avgpool = nn.AvgPool2d(kernel_size=(5,1),stride=(3,1),padding=(2,0))
            self.maxpool = nn.AdaptiveMaxPool1d(1)
            self.linear = nn.Linear(n_featmaps*3, 2)
            self.dropout = nn.Dropout(dropout_rate)
    #
        def forward(self, inputs): # inputs (bs,words/sentence) 10,7
            bsz = inputs.size(0) # batch size might change
            if inputs.size(1) < 3: # padding issues on really short sentences
                pads = Variable(torch.zeros(bsz,3-inputs.size(1))).type(torch.LongTensor)
                inputs = torch.cat([inputs,pads.cuda()],dim=1)
            embeds = self.embeddings(inputs) # 10,h,300
            out = embeds.unsqueeze(1) # 10,1,h,300
            out = out.permute(0,3,2,1) # 10,300,h,1
            fw3 = self.conv3(out) # 10,100,h,1
            fw5 = self.conv5(out) # 10,100,h,1
            fw7 = self.conv7(out) # 10,100,h,1
            out = torch.cat([fw3,fw5,fw7],dim=1)
            out = F.relu(out) # 10,300,h/3,1
            #out = self.avgpool(out)
            #out = F.relu(self.conv(out))
            out = out.view(bsz,n_featmaps*3,-1) # 10,300,7
            out = self.maxpool(out) # 10,300,1
            out = out.view(bsz,-1) # 10,300   
            out = self.dropout(out) # 10,2
            out = self.linear(out) # 10,2
            return out

    model = CNN()
# criterion = nn.CrossEntropyLoss() # accounts for the softmax component?
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

elif net_flag == 'multi':
    class MCNN(nn.Module):
        def __init__(self,second_glove):
            super(MCNN, self).__init__()
            self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
            self.embeddings.weight.data.copy_(word2vec)
            self.s_embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
            if second_glove == True:
                self.s_embeddings.weight.data.copy_(glove) # sasha uses the copy_
            else:
                self.s_embeddings.weight.data.copy_(word2vec)
            self.s_embeddings.weight.requires_grad = False
            self.conv = nn.Conv2d(n_featmaps*3,n_featmaps2,kernel_size=(filter_window,1),padding=(1,0))
            self.conv3 = nn.Conv2d(300,n_featmaps,kernel_size=(3,1),stride=(1,1),padding=(1,0))
            self.conv5 = nn.Conv2d(300,n_featmaps,kernel_size=(5,1),stride=(1,1),padding=(2,0))
            self.conv7 = nn.Conv2d(300,n_featmaps,kernel_size=(7,1),stride=(1,1),padding=(3,0))
            self.avgpool = nn.AvgPool2d(kernel_size=(5,2),stride=(3,2),padding=(2,0))
            self.maxpool = nn.AdaptiveMaxPool2d((1,2))
            self.linear = nn.Linear(n_featmaps*3*2, 2)
            self.dropout = nn.Dropout(dropout_rate)
    #
        def forward(self, inputs): # inputs (bs,words/sentence) 10,7
            bsz = inputs.size(0) # batch size might change
            if inputs.size(1) < 3: # padding issues on really short sentences
                pads = Variable(torch.zeros(bsz,3-inputs.size(1))).type(torch.LongTensor)
                inputs = torch.cat([inputs,pads.cuda()],dim=1)
            embeds = self.embeddings(inputs) # 10,h,300
            embeds = embeds.unsqueeze(3)
            embeds = embeds.permute(0,2,1,3)
            s_embeds = self.s_embeddings(inputs)
            s_embeds = s_embeds.unsqueeze(3)
            s_embeds = s_embeds.permute(0,2,1,3)
            out = torch.cat([embeds,s_embeds],dim=3)
            #print(out.size())
            fw3 = self.conv3(out) # 10,100,h,1
            fw5 = self.conv5(out) # 10,100,h,1
            fw7 = self.conv7(out) # 10,100,h,1
            out = torch.cat([fw3,fw5,fw7],dim=1)
            out = F.relu(out) # 10,300,h/3,1
            #out = self.avgpool(out)
            #out = F.relu(self.conv(out))
            #print(out.size())
            #out = out.view(bsz,n_featmaps*3,-1,2) # 10,300,7
            #print(out.size())
            out = self.maxpool(out) # 10,300,2
            out = out.view(bsz,-1) # 10,600   
            out = self.dropout(out) # 10,2
            out = self.linear(out) # 10,2
            return out

    is_glove = True

    model = MCNN(is_glove)

elif net_flag == 'extralayer':
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
            self.embeddings.weight.data = TEXT.vocab.vectors
            self.conv = nn.Conv2d(n_featmaps*3,n_featmaps2,kernel_size=(filter_window,1),padding=(1,0))
            self.conv3 = nn.Conv2d(300,n_featmaps,kernel_size=(3,1),padding=(1,0))
            self.conv5 = nn.Conv2d(300,n_featmaps,kernel_size=(5,1),padding=(2,0))
            self.conv7 = nn.Conv2d(300,n_featmaps,kernel_size=(7,1),padding=(3,0))
            self.avgpool = nn.AvgPool2d(kernel_size=(5,1),stride=(3,1),padding=(2,0))
            self.maxpool = nn.AdaptiveMaxPool1d(1)
            self.linear = nn.Linear(n_featmaps*2, 2)
            self.dropout = nn.Dropout(dropout_rate)
    #
        def forward(self, inputs): # inputs (bs,words/sentence) 10,7
            bsz = inputs.size(0) # batch size might change
            if inputs.size(1) < 3: # padding issues on really short sentences
                pads = Variable(torch.zeros(bsz,3-inputs.size(1))).type(torch.LongTensor)
                inputs = torch.cat([inputs,pads.cuda()],dim=1)
            embeds = self.embeddings(inputs) # 10,h,300
            out = embeds.unsqueeze(1) # 10,1,h,300
            out = out.permute(0,3,2,1) # 10,300,h,1
            fw3 = self.conv3(out) # 10,100,h,1
            fw5 = self.conv5(out) # 10,100,h,1
            fw7 = self.conv7(out) # 10,100,h,1
            out = torch.cat([fw3,fw5,fw7],dim=1)
            out = F.relu(out) # 10,300,h/3,1
            out = self.avgpool(out)
            out = F.relu(self.conv(out))
            out = out.view(bsz,n_featmaps2,-1) # 10,200,7
            out = self.maxpool(out) # 10,200,1
            out = out.view(bsz,-1) # 10,200   
            out = self.dropout(out) # 10,2
            out = self.linear(out) # 10,2
            return out

    model = CNN()

elif net_flag == 'dialated':
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
            self.embeddings.weight.data = TEXT.vocab.vectors
            self.conv = nn.Conv2d(n_featmaps*3,n_featmaps2,kernel_size=(filter_window,1),padding=(1,0),dialated=2)
            self.conv3 = nn.Conv2d(300,n_featmaps,kernel_size=(3,1),padding=(1,0))
            self.conv5 = nn.Conv2d(300,n_featmaps,kernel_size=(5,1),padding=(2,0))
            self.conv7 = nn.Conv2d(300,n_featmaps,kernel_size=(7,1),padding=(3,0))
            self.avgpool = nn.AvgPool2d(kernel_size=(5,1),stride=(3,1),padding=(2,0))
            self.maxpool = nn.AdaptiveMaxPool1d(1)
            self.linear = nn.Linear(n_featmaps*2, 2)
            self.dropout = nn.Dropout(dropout_rate)
    #
        def forward(self, inputs): # inputs (bs,words/sentence) 10,7
            bsz = inputs.size(0) # batch size might change
            if inputs.size(1) < 3: # padding issues on really short sentences
                pads = Variable(torch.zeros(bsz,3-inputs.size(1))).type(torch.LongTensor)
                inputs = torch.cat([inputs,pads.cuda()],dim=1)
            embeds = self.embeddings(inputs) # 10,h,300
            out = embeds.unsqueeze(1) # 10,1,h,300
            out = out.permute(0,3,2,1) # 10,300,h,1
            fw3 = self.conv3(out) # 10,100,h,1
            fw5 = self.conv5(out) # 10,100,h,1
            fw7 = self.conv7(out) # 10,100,h,1
            out = torch.cat([fw3,fw5,fw7],dim=1)
            out = F.relu(out) # 10,300,h/3,1
            out = self.avgpool(out)
            out = F.relu(self.conv(out))
            out = out.view(bsz,n_featmaps2,-1) # 10,200,7
            out = self.maxpool(out) # 10,200,1
            out = out.view(bsz,-1) # 10,200   
            out = self.dropout(out) # 10,2
            out = self.linear(out) # 10,2
            return out

    model = CNN()



# # Multichannel CNN
# class MCNN(nn.Module):
#     def __init__(self,second_glove):
#         super(MCNN, self).__init__()
#         self.embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
#         self.embeddings.weight.data.copy_(word2vec)
#         self.s_embeddings = nn.Embedding(TEXT.vocab.vectors.size(0),TEXT.vocab.vectors.size(1))
#         if second_glove == True:
#             self.s_embeddings.weight.data.copy_(glove) # sasha uses the copy_
#         else:
#             self.s_embeddings.weight.data.copy_(word2vec)
#             self.s_embeddings.weight.requires_grad = False
#         self.conv = nn.Conv2d(2,n_featmaps,kernel_size=(filter_window,300))
#         self.maxpool = nn.AdaptiveMaxPool1d(1)
#         self.linear = nn.Linear(n_featmaps, 2)
#         self.dropout = nn.Dropout(dropout_rate)
#     def forward(self, inputs): # inputs (bs,words/sentence) 10,7
#         bsz = inputs.size(0) # batch size might change
#         if inputs.size(1) < 3: # padding issues on really short sentences
#             pads = Variable(torch.zeros(bsz,3-inputs.size(1))).type(torch.LongTensor)
#             inputs = torch.cat([inputs,pads],dim=1)
#         embeds = self.embeddings(inputs) # 10,7,300
#         embeds = embeds.unsqueeze(1) # 10,1,7,300
#         s_embeds = self.s_embeddings(inputs) # 10,7,300
#         s_embeds = s_embeds.unsqueeze(1) # 10,1,7,300
#         out = torch.cat([embeds,s_embeds],dim=1) # 10,2,7,300
#         out = F.relu(self.conv(out)) # 10,100,6,1
#         out = out.view(bsz,n_featmaps,-1) # 10,100,6
#         out = self.maxpool(out) # 10,100,1
#         out = out.view(bsz,-1) # 10,100
#         out = self.linear(out) # 10,2
#         out = self.dropout(out) # 10,2
#         return out

# is_glove = True

# model = MCNN(is_glove)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss() # accounts for the softmax component?
params = filter(lambda x: x.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

losses = []

for epoch in range(num_epochs):
    train_iter.init_epoch()
    ctr = 0
    for batch in train_iter:
        sentences = batch.text.transpose(1,0)
        sentences = sentences.cuda()
        labels = (batch.label==1).type(torch.LongTensor)
        labels = labels.cuda()
        # change labels from 1,2 to 1,0
        optimizer.zero_grad()
        outputs = model(sentences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        nn.utils.clip_grad_norm(model.parameters(), constraint)
        ctr += 1
        if ctr % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                %(epoch+1, num_epochs, ctr, len(train)//bs, loss.data[0]))
        losses.append(loss.data[0])

    # normally you'd tab these back a level, but i'm paranoid
    np.save("../data/cnn_{}_losses".format(net_flag),np.array(losses))
    torch.save(model.state_dict(), '../data/cnn_{}.pkl'.format(net_flag))

# model.load_state_dict(torch.load('../../models/0cnn.pkl'))

model.eval() # lets dropout layer know that this is the test set
correct = 0
total = 0
for batch in val_iter:
    sentences = batch.text.transpose(1,0)
    sentences = sentences.cuda()
    labels = (batch.label==1).type(torch.LongTensor).data
    labels = labels.cuda()
    # change labels from 1,2 to 1,0
    outputs = model(sentences)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('test accuracy', correct/total)

def test(model):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    # test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        sentences = batch.text.transpose(1,0)
        sentences = sentences.cuda()
        probs = model(sentences)
        _, argmax = probs.max(1)
        upload += list(argmax.data)

    with open("../data/cnn_{}_predictions.csv".format(net_flag), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Id','Cat'])
        idcntr = 0
        for u in upload:
            if u == 0:
                u = 2
            writer.writerow([idcntr,u])
            idcntr += 1
            # f.write(str(u) + "\n")

test(model)


###############################

