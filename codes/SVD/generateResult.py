
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
import sys

def colorBars(words, token_list, svd_matrix, fileName):
    fig, axs = plt.subplots(nrows=len(words), ncols=1,
                            figsize=(20, 2*len(words)))
    color = ['brown', 'red', 'orange', 'yellow',
             'white', 'cyan', 'blue', 'purple', 'black']
    for i, word in enumerate(words):
        word_ind = token_list.index(word)
        vec = svd_matrix[word_ind]
        vec = vec/np.linalg.norm(vec)
        axs[i].bar(range(100), vec, color=color)
        axs[i].set_title(word)

    plt.tight_layout()
    plt.savefig(fileName)
    plt.show()


def plot_top_10_words(word, token_list, svd_matrix):
    def tsne_plot(results):
        words = [x[0] for x in results]
        tokens = [x[1] for x in results]
        words = np.array(words)
        tokens = np.array(tokens)

        tsne_model = TSNE(init='pca', perplexity=9, random_state=42)
        res_embeds = tsne_model.fit_transform(tokens)

        x_axis = res_embeds[:, 0]
        y_axis = res_embeds[:, 1]

        plt.figure(figsize=(10, 10))
        for i in range(len(x_axis)):
            plt.scatter(x_axis[i], y_axis[i])
            plt.annotate(words[i], xy=(x_axis[i], y_axis[i]), xytext=(
                5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.savefig('./Results/' + word+'.png')

    if word not in token_list:
        print("Word not in vocab")
        word = '<UNK>'
    word_index = token_list.index(word)
    word_vector = svd_matrix[word_index]
    res = {}
    for i, embed in tqdm(enumerate(svd_matrix)):
        if i != word_index:
            magnitude_product = np.linalg.norm(
                embed) * np.linalg.norm(word_vector)
            if magnitude_product == 0:
                res[i] = [np.nan, embed]
            else:
                cosine_distance = 1-distance.cosine(embed, word_vector)
            res[i] = [cosine_distance, embed]

    results = sorted(res.items(), key=lambda x: x[1][0], reverse=True)[:10]
    results = [(token_list[x[0]], x[1][1]) for x in results]

    print('Word: ', word)
    # print(results)
    print('Top 10 similar words: ', [x[0] for x in results])
    print_words = []
    print_words.append(word)
    for i in range(10):
        print_words.append(results[i][0])
    file_name = './Results/ '+word + "_bar.png"
    # colorBars(print_words, token_list, svd_matrix, file_name)
    # tsne_plot(results)


# get the SVD matrix
svd_matrix = []
feature_list = []
with open("embeddings.txt") as file:
    for line in file:
        svd_matrix.append([float(x) for x in line.strip('\n').split(" ")[1:]])
        feature_list.append(line.strip('\n').split(" ")[0])

svd_matrix = np.array(svd_matrix)

inputWord = input("Enter the word: ")
plot_top_10_words(inputWord, feature_list, svd_matrix)


