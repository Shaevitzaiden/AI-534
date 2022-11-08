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

def linearSVM_compare(X_train, y_train, X_dev, y_dev, i_s):
    cs = np.power(10*np.ones(i_s.shape), i_s)
    accuracy = []
    for i, c in enumerate(cs):
        quadSVM = SVC(C=c, kernel='linear')
        quadSVM.fit(X_train, y_train)
        predictions = quadSVM.predict(X_dev)
        accuracy.append(np.sum(predictions == y_dev)/np.size(y_dev))
        print("i = {0}, c = {1}, accuracy = {2}%".format(i_s[i], np.round(c,2), np.round(100*accuracy[i],3)))
    
    best_idx = np.argmax(accuracy)
    print("----- BEST -----")
    print("i = {0}, c = {1}, accuracy = {2}%".format(i_s[best_idx], np.round(cs[best_idx],2), np.round(100*accuracy[best_idx],3)))

    plt.plot(np.log10(cs), accuracy)
    plt.xlabel("$log_{10}$(c)")
    plt.ylabel("Accuracy (%)")
    plt.title("Linear SVM Accuracy vs c")
    plt.show()
    return accuracy

def quadraticSVM_compare(X_train, y_train, X_dev, y_dev, i_s):
    cs = np.power(10*np.ones(i_s.shape), i_s)
    accuracy = []
    for i, c in enumerate(cs):
        linSVM = SVC(C=c, kernel='poly', degree=2)
        linSVM.fit(X_train, y_train)
        predictions = linSVM.predict(X_dev)
        accuracy.append(np.sum(predictions == y_dev)/np.size(y_dev))
        print("i = {0}, c = {1}, accuracy = {2}%".format(i_s[i], np.round(c,2), np.round(100*accuracy[i],3)))
    
    best_idx = np.argmax(accuracy)
    print("----- BEST -----")
    print("i = {0}, c = {1}, accuracy = {2}%".format(i_s[best_idx], np.round(cs[best_idx],2), np.round(100*accuracy[best_idx],3)))

    plt.plot(np.log10(cs), accuracy)
    plt.xlabel("$log_{10}$(c)")
    plt.ylabel("Accuracy (%)")
    plt.title("Quadratic SVM Accuracy vs c")
    plt.show()
    return accuracy

def rbfSVM_compare(X_train, y_train, X_dev, y_dev, ics, igs):
    cs = np.power(10*np.ones(ics.shape), ics)
    gs = np.power(10*np.ones(igs.shape), igs)
    cc, gg = np.meshgrid(cs, gs)
    cc_vertical = cc.flatten().reshape(cc.size,1)
    gg_vertical = gg.flatten().reshape(gg.size,1)
    
    # Create vector of combinations so we can use a single loop as opposed to nested loops, just less messy
    combo_vector = np.hstack((cc_vertical, gg_vertical))
    print(combo_vector)
    
    accuracy = []
    for i, (g, c) in enumerate(combo_vector):
        rbfSVM = SVC(C=c, kernel='rbf', gamma=g)
        rbfSVM.fit(X_train, y_train)
        predictions = rbfSVM.predict(X_dev)
        accuracy.append(np.sum(predictions == y_dev)/np.size(y_dev))
        print("g = {0}, c = {1}, accuracy = {2}%".format(np.round(g,2), np.round(c,2), np.round(100*accuracy[i],3)))
    
    best_idx = np.argmax(accuracy)
    print("----- BEST -----")
    best_g, best_c = combo_vector[best_idx]
    print("g = ({0}, {1}), c = ({2}, {3}), accuracy = {4}%".format(np.round(best_g,2), np.round(np.log10(best_g),2), np.round(best_c,2), np.round(np.log10(best_c),2), np.round(100*accuracy[best_idx],3)))

    # plt.plot(np.log10(cs), accuracy)
    # plt.xlabel("$log_{10}$(c)")
    # plt.ylabel("Accuracy (%)")
    # plt.title("Quadratic SVM Accuracy vs c")
    # plt.show()
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

    ic_base = np.arange(-1, 2, 1)
    ic_fine = np.arange(-1, 1.1, 0.1,)
    
    ig_base = np.arange(-1, 2, 1)
    ig_fine = None

    # lin_accuracies = linearSVM_compare(X_train, y_train, X_dev, y_dev, ic_base)
    # quad_accuracies = quadraticSVM_compare(X_train, y_train, X_dev, y_dev, ic_base)
    rbf_accuracies = rbfSVM_compare(X_train, y_train, X_dev, y_dev, ic_base, ig_base)


     