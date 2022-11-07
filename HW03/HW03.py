#!/usr/bin/env python3

import sklearn as sk
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC


def preprocess(data_train, data_dev, method="TF-IDF"):
    if method == "TF-IDF":
        vectorizer = TfidfVectorizer(use_idf=True)
    elif method == "Vectorizer":
        vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(data_train[:,1])
    X_dev = vectorizer.transform(data_dev[:,1])
    names = vectorizer.get_feature_names_out()
    return X_train, X_dev, names
        
def load_data(path):
    """
    Loads and pre-processes data
    :param path: Path to data
    :return: sentiment vector, sentence vector
    """
    # Load data and seperate data into numpy array and headers
    data_pd = pd.read_csv(path)
    data_np = data_pd.to_numpy()
    # sentiment = data_np[:,0]
    # sentences = data_np[:,1]

    return data_np

def linearSVM_compare(X_train, y_train, X_dev, y_dev, cs):
    accuracy = []
    for i, c in enumerate(cs):
        linSVM = SVC(C=c, kernel='linear')
        linSVM.fit(X_train, y_train)
        predictions = linSVM.predict(X_dev)
        accuracy.append(np.sum(predictions == y_dev)/np.size(y_dev))
        print("c = {0}, accuracy = {1}%".format(c, np.round(100*accuracy[i],2)))
        
        # print(confusion_matrix(y,predictions))
        # print(classification_report(y,predictions))
    plt.plot(cs, accuracy)
    plt.xlabel("c-value")
    plt.ylabel("Accuracy (%)")
    plt.title("Linear SVM Accuracy vs c")
    return accuracy


if __name__ == "__main__":
    data_train = load_data("HW03\IA3-train.csv")
    data_dev = load_data("HW03\IA3-dev.csv")

    """ top 10 stuff
    # count_vectors = X.toarray()
    # summed_counts = np.sum(count_vectors, axis=0)
    # idxs = np.argpartition(summed_counts, -10)[-10:]
    
    # print(summed_counts[idxs]) # counts
    # print(names[idxs]) # highest word counts

    # pos_sentences = sentences[sentiment == 1]
    # neg_sentences = sentences[sentiment == 0]

    # vectorizer = TfidfVectorizer(use_idf=True)
    # X_pos = vectorizer.fit_transform(pos_sentences)
    # names_pos = vectorizer.get_feature_names_out()

    # X_neg = vectorizer.fit_transform(neg_sentences)
    # names_neg = vectorizer.get_feature_names_out()
    
    # idf_pos_vectors = X_pos.toarray()
    # idf_neg_vectors = X_neg.toarray()

    # summed_idfs_pos = np.sum(idf_pos_vectors, axis=0)
    # summed_idfs_neg = np.sum(idf_neg_vectors, axis=0)
    
    # idxs_pos = np.argpartition(summed_idfs_pos, -10)[-10:]
    # idxs_neg = np.argpartition(summed_idfs_neg, -10)[-10:]
    
    
    # # print(summed_idfs_pos[idxs_pos])
    # print(names_pos[idxs_pos]) # highest word counts

    # # print(summed_idfs_neg[idxs_neg])
    # print(names_neg[idxs_neg]) # highest word counts
    """

    X_train, X_dev, words_train = preprocess(data_train, data_dev)
    
    y_train = data_train[:,0].astype(int)
    y_dev = data_dev[:,0].astype(int)

    i = np.arange(start=-4, stop=5, step=1)
    c = np.power(10*np.ones(i.shape), i)

    lin_accuracies = linearSVM_compare(X_train, y_train, X_dev, y_dev, c)
     