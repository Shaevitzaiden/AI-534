#!/usr/bin/env python3

import sklearn as sk
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def preprocess(data, method="TF-IDF"):
    if method == "TF-IDF":
        pass
    elif method == "Vectorizer":
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)
        s = vectorizer.get_feature_names_out()
        
        count_vectors = X.toarray()
        summed_counts = np.sum(count_vectors, axis=0)
        idxs = np.argpartition(summed_counts, -10)[-10:]
        
        print(summed_counts[idxs]) # counts
        print(s[idxs]) # highest word counts
        

def load_data(path):
    """
    Loads and pre-processes data
    :param path: Path to data
    :return: sentiment vector, sentence vector
    """
    # Load data and seperate data into numpy array and headers
    data_pd = pd.read_csv(path)
    data_np = data_pd.to_numpy()
    sentiment = data_np[:,0]
    sentences = data_np[:,1]

    return sentiment, sentences

if __name__ == "__main__":
    sentiment, sentences = load_data("HW03\IA3-train.csv")

    pos_sentences = sentences[sentiment == 1]
    neg_sentences = sentences[sentiment == 0]

    vectorizer = TfidfVectorizer()
    X_pos = vectorizer.fit_transform(pos_sentences)
    names_pos = vectorizer.get_feature_names_out()

    X_neg = vectorizer.fit_transform(neg_sentences)
    names_neg = vectorizer.get_feature_names_out()
    
    idf_pos_vectors = X_pos.toarray()
    idf_neg_vectors = X_neg.toarray()

    summed_idfs_pos = np.sum(idf_pos_vectors, axis=0)
    summed_idfs_neg = np.sum(idf_neg_vectors, axis=0)
    
    idxs_pos = np.argpartition(summed_idfs_pos, -10)[-10:]
    idxs_neg = np.argpartition(summed_idfs_neg, -10)[-10:]
    
    print(summed_idfs_pos[idxs_pos])
    print(names_pos[idxs_pos]) # highest word counts

    print(summed_idfs_neg[idxs_neg])
    print(names_neg[idxs_neg]) # highest word counts
    
    # print(summed_counts[idxs]) # counts
    # print(names[idxs]) # highest word counts
