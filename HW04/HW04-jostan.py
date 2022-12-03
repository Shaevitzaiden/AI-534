import numpy as np
import sklearn.metrics

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from GloVe_Embedder import GloVe_Embedder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sklearn.metrics

def get_top_words(ge, seed_words, num_neighbors):
    nearest_neighbors = []
    for i, word in enumerate(seed_words):
        nearest_neighbors.append(ge.find_k_nearest(word, num_neighbors))
    return nearest_neighbors

def print_top_words(nearest_neighbors, formatted=False):
    if not formatted:
        for word_list in nearest_neighbors:
            for word in word_list:
                print(word[0] + ": " + str(round(word[1], 2)))
            print()

    if formatted:
        for i in range(len(nearest_neighbors[0])):
            print_str = ""
            for word in nearest_neighbors:
                print_str += word[i][0] + " & " + str(round(word[i][1], 2)) + " & "
            print_str = print_str[:-2]
            print_str += "\\\\"
            print(print_str)
            print("\\hline")

def plot_points(x, y):
    fig, ax = plt.subplots()
    start = 0
    colors = ["b", "g", "r", "k", "y"]
    for i in range(5):

        ax.scatter(x[start:start+30], y[start:start+30], color=colors[i])
        start += 30

    # fig.show()

def compare_tsne(embeddings):
    perplexities = [5, 10, 15, 20, 25, 30, 35, 40, 50]
    p = 0
    fig, axs = plt.subplots(3, 3)
    colors = ["b", "g", "r", "k", "y"]
    for i in range(3):
        for j in range(3):
            z = TSNE(n_components=2, perplexity=perplexities[p]).fit_transform(embeddings.copy())
            x, y = z[:, 0], z[:, 1]
            start = 0
            for s in range(5):
                axs[i,j].scatter(x[start:start+30], y[start:start+30], color=colors[s])
                start += 30
            axs[i,j].set_title('Perplexity = {0}'.format(perplexities[p]))
            p += 1


if __name__ == "__main__":
    seed_words = ('flight', 'good', 'terrible', 'help', 'late')
    ge = GloVe_Embedder("GloVe_Embedder_data.txt")
    top_words = get_top_words(ge, seed_words, 30)
    # print_top_words(top_words, formatted=True)

    # Get full list of just words
    words = []
    for word_list in top_words:
        for word in word_list:
            words.append(word[0])

    # Get embedings for list of words
    embedings = ge.embed_list(words)

    # Do PCA and plot it
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(embedings)
    # plot_points(principalComponents[:, 0], principalComponents[:, 1])

    # Do tsne and plot it
    compare_tsne(embedings.copy())
    # plt.show()

    # Do kmeans
    inertias = []
    ns = [n for n in range(2,21)]
    fig, ax = plt.subplots()
    label_ground_truth = ([n for n in range(5) for i in range(30)])
    ari = []
    nmi = []
    purity = []
    for n in ns:
        km = KMeans(n_clusters=n)
        y_km = km.fit_transform(embedings)
        inertias.append(km.inertia_)
        label_pred = km.labels_
        ari.append(sklearn.metrics.adjusted_rand_score(label_ground_truth, label_pred))
        nmi.append(sklearn.metrics.normalized_mutual_info_score(label_ground_truth, label_pred))
        # Calculate purity
        max_class_in_cluster_sum = 0
        for i in range(n):
            start = 0
            num_class_in_cluster = []
            for j in range(5):
                num_class_in_cluster.append(sum(label_pred[start:start + 30] == i))
                start += 30

            max_class_in_cluster_sum += (max(num_class_in_cluster))
        purity.append(max_class_in_cluster_sum / 150)

    # Plot inertiias
    ax.plot(ns, inertias)
    ax.set_xticks(ns)
    ax.set_ylim(0, 3000)
    # fig.show()







