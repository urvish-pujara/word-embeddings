import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from tqdm import tqdm 
from torch.nn import init
import random
import gensim
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse


np.random.seed(21)

class Data:
    TABLE_SIZE = 1e8
    def __init__(self,training_data,min_freq,to_be_discard_size):
        self.training_data = training_data
        self.min_freq = min_freq
        self.to_be_discard_size = to_be_discard_size
        self.word2index = {}
        self.index2word = {}
        self.word_freq = {}
        self.sentence_count = 0 
        self.words_count = 0
        self.vocab_size = 0
        self.negatives = []
        self.subSampleTable = []
        self.negativePos = 0
        self.generateWords()
        self.generateNegatives()
        self.generateSubSampleTable()
        

    def generateWords(self):
        temp_dict = {}
        for sentence in self.training_data:
            sentence = sentence.split()
            if len(sentence) < 2:
                continue
            self.sentence_count += 1
            for word in sentence:
                if len(word) < 1:
                    continue
                self.words_count += 1
                if word not in temp_dict:
                    temp_dict[word] = 1
                else:
                    temp_dict[word] += 1
                if self.words_count %(1e6) == 0:
                    print("Processed {} M tokens".format(self.words_count))
        # indexing words for words , start, end and padding
        self.word2index = {'<PAD>':0,'<S>':1,'</S>':2}
        self.index2word = {0:'<PAD>',1:'<S>',2:'</S>'}
        self.word_freq = {'<PAD>':1,'<S>':1,'</S>':1}
        for word,cnt in temp_dict.items():
            # if cnt >= self.min_freq:
            self.word2index[word] = len(self.word2index)
            self.index2word[len(self.index2word)] = word
            self.word_freq[word] = cnt

        self.vocab_size = len(self.word2index)
        print("Vocab size is {}".format(self.vocab_size))
        print("Total number of words {}".format(self.words_count))
        print("Total number of sentences {}".format(self.sentence_count))

    def generateNegatives(self):
        newFreqWords = []
        for word in self.word_freq:
            newFreqWords.append(self.word_freq[word])
        newFreqWords = np.array(newFreqWords)**0.75
        newFreqWords = newFreqWords/np.sum(newFreqWords)
        self.negatives = np.random.choice(len(newFreqWords),size=int(Data.TABLE_SIZE),p=newFreqWords)
        print("Negative samples generated")

    def generateSubSampleTable(self):
        # f = sqrt(t/f) + t/f
        fr = np.array(list(self.word_freq.values()))
        fr = fr/np.sum(fr)
        fr = np.sqrt(1e-3/fr) + 1e-3/fr
        self.subSampleTable = np.round(fr * self.TABLE_SIZE)
        print("Subsample table generated")

    def getNegativeSamples(self,target):
        negSamples = []
        while len(negSamples) < self.to_be_discard_size:
            neg = self.negatives[self.negativePos]
            self.negativePos = (self.negativePos + 1) % len(self.negatives)
            if neg not in negSamples and neg != target:
                negSamples.append(neg)

        if len(negSamples) != self.to_be_discard_size:
            return np.concatenate((negSamples,self.negatives[:self.to_be_discard_size-len(negSamples)]))
        return negSamples

class Word2vecDataLoader(Dataset):
    def __init__(self, data, windowSize):
        self.data = data
        self.windowSize = windowSize
        self.word2index = data.word2index
        self.index2word = data.index2word
        self.word_freq = data.word_freq
        self.subSampleTable = data.subSampleTable
        self.negatives = data.negatives
        self.length = data.sentence_count

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        sentence = self.data.training_data[idx].split()
        if len(sentence) < 2:
            # Can not return none as it will throw error
            if idx+1 == self.length:
                idx = 0
            return self.__getitem__(idx+1)
        sentence_ids = [self.word2index[word] for word in sentence if word in self.word2index and np.random.rand() < self.subSampleTable[self.word2index[word]]]
        padded_sent = [self.word2index['<S>']] + sentence_ids + [self.word2index['</S>']]
        cbowData = []
        szOneSide = self.windowSize//2
        for ind,word in enumerate(padded_sent):
            if ind == 0 or ind == len(padded_sent)-1:
                continue
            start = max(0,ind-szOneSide)
            end = min(len(padded_sent)-1,ind+szOneSide)
            context = [padded_sent[i] for i in range(start,end+1) if i != ind]
            if len(context) < 2*(szOneSide):
                context += [self.word2index['<PAD>']]*(2*(szOneSide)-len(context))
            
            cbowData.append((context,word,self.data.getNegativeSamples(word)))

        return cbowData
    
    @staticmethod
    def collate_fn(batches):
        targ_batch = [u for batch in batches for u,v,n in batch if len(batch) > 0]
        context_batch = [v for batch in batches for u,v,n in batch if len(batch) > 0]
        neg_batch = [n for batch in batches for u,v,n in batch if len(batch) > 0]

        return torch.LongTensor(context_batch),torch.LongTensor(targ_batch),torch.LongTensor(neg_batch)
    

class CBOWModule(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(CBOWModule, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.targetEmbedding = nn.Embedding(
            vocab_size, embedding_size, sparse=True)
        self.contextEmbedding = nn.Embedding(
            vocab_size, embedding_size, sparse=True)
        initrange = 1.0/self.embedding_size
        init.uniform_(self.targetEmbedding.weight.data, -initrange, initrange)
        init.constant_(self.contextEmbedding.weight.data, 0)

    def forward(self, context, target, negatives):
        trg_embedding = self.targetEmbedding(target)
        trg_embedding = torch.mean(trg_embedding, dim=1)
        context_embedding = self.contextEmbedding(context)
        neg_embedding = self.contextEmbedding(negatives)

        positive_score = torch.sum(
            torch.mul(trg_embedding, context_embedding), dim=1)
        positive_score = torch.clamp(positive_score, max=10, min=-10)
        positive_score = -F.logsigmoid(positive_score)

        neg_score = torch.bmm(
            neg_embedding, torch.unsqueeze(trg_embedding, 2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(positive_score+neg_score)

    def embeddingSave(self, ind2word, output_file):
        embedding = self.targetEmbedding.weight.data.cpu().numpy()
        with open(output_file, 'w') as f:
            f.write('{} {}\n'.format(len(ind2word), self.embedding_size))
            for word_id, word in ind2word.items():
                tmp = ' '.join(map(lambda x: str(x), embedding[word_id]))
                f.write('{} {}\n'.format(word, tmp))



EMBEDDING_DIM = 300
BATCH_SIZE = 128
WINDOW_SIZE = 11
EPOCHS = 10
LEARNING_RATE = 0.001
NEGATIVE_SAMPLES = 5
MIN_FREQ = 5

class CBOW:
    def __init__(self,training_data,output_file):
        self.data = Data(training_data,min_freq=MIN_FREQ,to_be_discard_size=NEGATIVE_SAMPLES)
        compData = Word2vecDataLoader(self.data,WINDOW_SIZE)
        self.dataloader = DataLoader(compData,batch_size=BATCH_SIZE,shuffle=True,collate_fn=compData.collate_fn)
        self.embedding_size = len(self.data.word2index)
        self.embedding_dim = EMBEDDING_DIM
        self.model = CBOWModule(self.embedding_size, self.embedding_dim)
        self.use_cuda = torch.cuda.is_available()
        self.output_file = output_file
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        optimizer = optim.SparseAdam(self.model.parameters(),lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,len(self.dataloader),eta_min=0.0001)
        for epoch in range(EPOCHS):
            total_loss = 0.0
            for i,(context,target,negatives) in enumerate(tqdm(self.dataloader)):
                if len(context) == 0:
                    continue
                cont = context.to(self.device)
                targ = target.to(self.device)
                neg = negatives.to(self.device)
                optimizer.step()
                optimizer.zero_grad()
                loss = self.model.forward(cont,targ,neg)
                loss.backward()
                scheduler.step()

                total_loss = total_loss*0.9 + loss.item()*0.1
                if i > 0 and i % 100 == 0:
                    print('Epoch: {} | Batch: {} | Loss: {}'.format(epoch,i,total_loss))
            print('Epoch: {} | Loss: {}'.format(epoch,total_loss))
        self.model.embeddingSave(self.data.index2word,self.output_file)


# dataLines = []
# with open("../../Data/preProcess.txt") as file:
#     for line in tqdm(file):
#         if line == " ":
#             continue
#         dataLines.append(line)

# print("Number of lines: ", len(dataLines))

# output_file = 'embeddings.txt'

# cbow = CBOW(dataLines[:200000], output_file)
# cbow.train()
