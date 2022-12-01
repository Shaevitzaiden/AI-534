#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GloVe_Embedder import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


if __name__ == "__main__":
    embedder = GloVe_Embedder("HW04/GloVe_Embedder_data.txt")
    
    # Part 1a - Build your own data set of words
    seed_words = ["flight", "good", "terrible", "help", "late"]
    num_similar_words = 29

    # Getting the euclidean distance for different seed words
    distance = np.zeros((num_similar_words, len(seed_words)))
    for i, word in enumerate(seed_words):
        similar_words = embedder.find_k_nearest(word, num_similar_words)
        for row in range(len(similar_words)):
            distance[row, i] = similar_words[row][1]
    

    # Part 1b - Dimension reduction and visualization
    pca = PCA()
    pca.fit(distance)



    # Part 1c - Clustering
    k = range(2, 20)
    for i in k:
        KMeans(n_clusters=k).fit(distance)
    

 


