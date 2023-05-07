import gensim
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import numpy as np  


def colorBars(words, cbow_own, fileName):
    fig, axs = plt.subplots(nrows=len(words), ncols=1,
                            figsize=(20, 2*len(words)))
    color = ['brown', 'red', 'orange', 'yellow',
             'white', 'cyan', 'blue', 'purple', 'black']
    for i, word in enumerate(words):
        vec = cbow_own[word]
        vec = vec/np.linalg.norm(vec)
        axs[i].bar(range(300), vec, color=color)
        axs[i].set_title(word)

    plt.tight_layout()
    plt.savefig(fileName)
    plt.show()


def plot_top10_words(word, cbow_own):

    def tsne_plot(results):
        words = [x[0] for x in results]
        embeds = [x[1] for x in results]
        words = np.array(words)
        embeds = np.array(embeds)
        tsne_model = TSNE(init='pca', perplexity=19, random_state=42)
        res_embeds = tsne_model.fit_transform(embeds)
        x_axis = res_embeds[:, 0]
        y_axis = res_embeds[:, 1]

        plt.figure(figsize=(10, 10))
        for i in range(len(x_axis)):
            plt.scatter(x_axis[i], y_axis[i])
            plt.annotate(words[i], xy=(x_axis[i], y_axis[i]), xytext=(
                5, 2), textcoords='offset points', ha='right', va='bottom')

    if word not in cbow_own:
        print('Word not in vocabulary')
        word = "<UNK>"
    similar_words = cbow_own.most_similar(positive=[word], topn=10)
    res = {}
    for i, embed in tqdm(enumerate(similar_words)):
        res[embed[0]] = [embed[1], cbow_own[embed[0]]]

    results = []
    scores = 0
    for t in sorted(res.items(), key=lambda item: item[1][0], reverse=True)[0:10]:
        results.append([t[0], t[1][1]])

    print('Word: ', word)
    print('Top 10 similar words: ', [x[0] for x in results])
    print_words = []
    print_words.append(word)
    for i in range(10):
        print_words.append(results[i][0])
    file_name = "./Results/" + word + "_bar.png"
    # colorBars(print_words,cbow_own,file_name)
    # tsne_plot(results)
    # return results


cbow_own = gensim.models.KeyedVectors.load_word2vec_format(
    'embeddings.txt', binary=False)


inputWord = input("Enter the word: ")
plot_top10_words(inputWord, cbow_own)
