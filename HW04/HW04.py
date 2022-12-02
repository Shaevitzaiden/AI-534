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
    distance = []
    for word in seed_words:
        nearest = embedder.find_k_nearest(word, num_similar_words)
        w = [x[0] for x in nearest]
        d = [x[1] for x in nearest]
        words.append(w)
        distance.append(d)
   
    embeddings = embedder.embed_list(words[0])
    # print(embeddings)

    # Part 1b - Dimension reduction and visualization
    pca = PCA(n_components=2)
    pca.fit_transform(embeddings)



    # # Part 1c - Clustering
    # k = range(2, 20)
    # for i in k:
    #     KMeans(n_clusters=k).fit(distance)
    

 


