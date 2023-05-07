from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm
import json
from collections import Counter
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
# lemmatisation
from nltk.stem import WordNetLemmatizer


from nltk.tokenize import word_tokenize


class SVD:
    def __init__(self, window_size, cutoff, data):
        self.data = data
        self.window_size = window_size
        self.cutoff = cutoff
        self.wordDict = defaultdict(int)
        self.wordCount = 0
        self.minfreq = 200
        self.word2id = {}
        self.id2word = {}
        self.vocab = []
        self.vocabSize = 0
        self.cooccur = defaultdict(int)
        self.cooccurMatrix = None
        self.wordTokens = []
        self.U = None
        self.S = None
        self.V = None
        self.dim = 0
        self.createVocab()
        self.createCooccurMatrix()
        self.createSVD()
        self.saveEmbeddings()

    def createVocab(self):
        for sent in tqdm(self.data):
            words = sent.split()
            self.wordTokens.append(words)
            for word in sent.split():
                self.wordDict[word] += 1
                self.wordCount += 1
            if self.wordCount % 1000000 == 0:
                print("Number of words processed: ", self.wordCount)

        self.vocab = list(self.wordDict.keys())
        print("Vocab size: ", len(self.vocab))

        with open("wordTokens.txt", "w") as f:
            for sent in self.wordTokens:
                f.write(" ".join(sent) + "\n")

        print("Tokens created")
        self.vocabSize = len(self.vocab)
        self.cooccurMatrix = sparse.lil_matrix(
            (self.vocabSize, self.vocabSize))

        with open("vocab.txt", "w") as f:
            for ind, word in enumerate(self.vocab):
                self.word2id[word] = ind
                self.id2word[ind] = word
                f.write(word + " " + str(self.wordDict[word]) + "\n")
        print("Vocab created")

    def createCooccurMatrix(self):
        with open("wordTokens.txt", "r") as f:
            for line in tqdm(f):
                line = line.split()
                first_window = line[:self.window_size]
                if self.window_size > len(line):
                    continue
                for t, word in enumerate(line[:-self.window_size]):
                    word_ind = self.word2id[word]
                    first_window = first_window[1:] + \
                        [line[t+self.window_size]]
                    for context_word in first_window:
                        context_ind = self.word2id[context_word]
                        try:
                            self.cooccurMatrix[word_ind, context_ind] += 1
                        except:
                            self.cooccurMatrix[word_ind, context_ind] = 1
                        try:
                            self.cooccurMatrix[context_ind, word_ind] += 1
                        except:
                            self.cooccurMatrix[context_ind, word_ind] = 1

        print("Cooccurrence matrix created")

    def createSVD(self):
        u, s, v = sparse.linalg.svds(self.cooccurMatrix, which='LM', k=100)
        print(u.shape, s.shape, v.shape)
        deno = sum(s)
        sum_count = 0
        # maximum variance cutoff
        for i, x in enumerate(s):
            sum_count += x
            if sum_count/deno > self.cutoff:
                self.dim = i+1
                break
        self.U = u[:, :self.dim]
        self.S = s[:self.dim]
        self.V = v[:self.dim, :]

        # normalise
        norms_u = np.linalg.norm(self.U, axis=1, keepdims=True)
        self.U = np.where(norms_u < 1e-8, self.U, self.U / (norms_u + 1e-8))

        norms_v = np.linalg.norm(self.V, axis=1, keepdims=True)
        self.V = np.where(norms_v < 1e-8, self.V, self.V / (norms_v + 1e-8))

        print("SVD done")

    def saveEmbeddings(self):
        with open("embeddings.txt", "w") as f:
            for word, vec in zip(self.vocab, self.U):
                f.write(word + " " + " ".join([str(x) for x in vec]) + "\n")







