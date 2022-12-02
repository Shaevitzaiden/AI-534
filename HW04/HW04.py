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
    words = []
    distances = []
    for word in seed_words:
        nearest = embedder.find_k_nearest(word, 5)
        w = [x[0] for x in nearest]
        d = [x[1] for x in nearest]
        words.append(w)
        distances.append(d)
    print(words)
    

    # Part 1b - Dimension reduction and visualization
    pca = PCA()
    pca.fit(distances)



    # Part 1c - Clustering
    k = range(2, 20)
    for i in k:
        KMeans(n_clusters=k).fit(distances)
    

 


