import numpy as np
from tqdm import tqdm
import json
from collections import Counter
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer


dataLines = []
with open("../../Data/reviews.txt") as file:
    for line in tqdm(file):
        if line == " ":
            continue
        dataLines.append(line)

print("Number of lines: ", len(dataLines))


lemma = WordNetLemmatizer()
def lemmatize(dataLines):
    newData = []
    for line in tqdm(dataLines):
        words = line.split()
        for i in range(len(words)):
            words[i] = lemma.lemmatize(words[i])
        newData.append(" ".join(words))
    return newData


def preProcessing(dataLines,minFreq):
    dataLines = lemmatize(dataLines)
    wordCount = defaultdict(int)

    for line in tqdm(dataLines):
        for word in line.split():
            wordCount[word] += 1
    newData = []
    for line in tqdm(dataLines):
        words = line.split()
        for i in range(len(words)):
            if wordCount[words[i]] < minFreq:
                words[i] = '<UNK>'
        newData.append(" ".join(words))
    # save the dataLines in review.txt
    with open("../../Data/preProcess.txt", "w") as file:
        for line in newData:
            file.write(line + "\n")

    return newData
