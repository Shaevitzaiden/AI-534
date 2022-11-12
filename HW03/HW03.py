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


    return data_np

def linearSVM_compare(X_train, y_train, X_dev, y_dev, i_s, sv_plot=False):
    cs = np.power(10*np.ones(i_s.shape), i_s)
    accuracy_test = []
    accuracy_train = []
    num_sv = []
    for i, c in enumerate(cs):
        linSVM = SVC(C=c, kernel='linear')
        linSVM.fit(X_train, y_train)
        predictions_test = linSVM.predict(X_dev)
        predictions_train = linSVM.predict(X_train)
        accuracy_test.append(np.sum(predictions_test == y_dev)/np.size(y_dev))
        accuracy_train.append(np.sum(predictions_train == y_train)/np.size(y_train))
        num_sv.append(linSVM.support_.size)
        print("i = {0}, c = {1}, accuracy = {2}%".format(i_s[i], np.round(c,2), np.round(100*accuracy_test[i],3)))
    
    best_idx = np.argmax(accuracy_test)
    print(num_sv)
    print("----- BEST -----")
    print("i = {0}, c = {1}, accuracy = {2}%".format(i_s[best_idx], np.round(cs[best_idx],2), np.round(100*accuracy_test[best_idx],3)))

    plt.plot(np.log10(cs), accuracy_test, label='Validation')
    plt.plot(np.log10(cs), accuracy_train, label='Train')
    plt.xlabel("$log_{10}$(c)")
    plt.ylabel("Accuracy")
    plt.title("Linear SVM Accuracy vs c")
    plt.legend()
    plt.show()

    if sv_plot:
        plt.clf()
        plt.plot(np.log10(cs), num_sv)
        plt.xlabel("$log_{10}$(c)")
        plt.ylabel("Number of Support Vectors")
        plt.title("Number of Support Vectors for Varying C-Values")
        plt.show()

    return accuracy_test

def quadraticSVM_compare(X_train, y_train, X_dev, y_dev, i_s, sv_plot=False):
    cs = np.power(10*np.ones(i_s.shape), i_s)
    accuracy_test = []
    accuracy_train = []
    num_sv = []
    for i, c in enumerate(cs):
        quadSVM = SVC(C=c, kernel='poly', degree=2)
        quadSVM.fit(X_train, y_train)
        predictions = quadSVM.predict(X_dev)
        predictions_train = quadSVM.predict(X_train)
        accuracy_test.append(np.sum(predictions == y_dev)/np.size(y_dev))
        accuracy_train.append(np.sum(predictions_train == y_train) / np.size(y_train))
        num_sv.append(quadSVM.support_.size)
        print("i = {0}, c = {1}, accuracy = {2}%".format(i_s[i], np.round(c,2), np.round(100*accuracy_test[i],3)))
    
    best_idx = np.argmax(accuracy_test)
    print("----- BEST -----")
    print("i = {0}, c = {1}, accuracy = {2}%".format(i_s[best_idx], np.round(cs[best_idx],2), np.round(100*accuracy_test[best_idx],3)))

    plt.plot(np.log10(cs), accuracy_test, label='Validation')
    plt.plot(np.log10(cs), accuracy_train, label='Train')
    plt.xlabel("$log_{10}$(c)")
    plt.ylabel("Accuracy")
    plt.title("Quadratic SVM Accuracy vs c")
    plt.legend()
    plt.show()

    if sv_plot:
        plt.clf()
        plt.plot(np.log10(cs), num_sv)
        plt.xlabel("$log_{10}$(c)")
        plt.ylabel("Number of Support Vectors")
        plt.title("Number of Support Vectors for Varying C-Values")
        plt.show()

    return accuracy_test

def rbfSVM_compare(X_train, y_train, X_dev, y_dev, ics, igs, heatmap=True, sv_plot=False):
    cs = np.power(10*np.ones(ics.shape), ics)
    gs = np.power(10*np.ones(igs.shape), igs)
    accuracy_train = np.zeros((gs.size, cs.size))
    accuracy_dev = accuracy_train.copy()
    sv_c_constant = []
    sv_g_constant = []

    for col, c in enumerate(cs): # columns
        for row, g in enumerate(gs): # rows
            rbfSVM = SVC(C=c, kernel='rbf', gamma=g)
            rbfSVM.fit(X_train, y_train)
            
            predictions_train = rbfSVM.predict(X_train)
            predictions_dev = rbfSVM.predict(X_dev)
            
            accuracy_train[row,col]= (np.sum(predictions_train == y_train)/np.size(y_train))
            accuracy_dev[row,col]= (np.sum(predictions_dev == y_dev)/np.size(y_dev))
            print("c = {0}, g = {1}, accuracy = {2}%".format(np.round(c,2), np.round(g,2), np.round(100*accuracy_dev[row,col],3)))
            if np.isclose(c, 10):
                sv_c_constant.append(rbfSVM.support_.size)
            if np.isclose(g, 0.1):
                sv_g_constant.append(rbfSVM.support_.size)

    br, bc = np.unravel_index(accuracy_dev.argmax(), accuracy_dev.shape)
    print("------- BEST -------")
    print("c = {0}, g = {1}, accuracy = {2}%".format(np.round(cs[bc],2), np.round(gs[br],2), np.round(100*accuracy_dev[br,bc],3)))

    if heatmap:
        f,(ax1,ax2, cax) = plt.subplots(1,3, gridspec_kw={'width_ratios': [1,1,0.1], 'height_ratios': [1]})
        min_val = np.min(np.hstack((accuracy_train, accuracy_train)))
        
        m1 = sns.heatmap(accuracy_train, cmap='hot', ax=ax1, cbar=False, vmin=min_val, vmax=1)
        m1.set_title('Training Accuracy')
        m1.set_xlabel("$log_{10}$(c)")
        m1.set_ylabel("$log_{10}$($\\gamma$)")
        m1.set_xticklabels(ics)
        m1.set_yticklabels(igs)

        m2 = sns.heatmap(accuracy_dev, cmap='hot', ax=ax2, cbar_ax=cax, vmin=min_val, vmax=1)
        m2.set_title('Validation Accuracy')
        m2.set_xlabel("$log_{10}$(c)")
        m2.set_ylabel("$log_{10}$($\\gamma$)")
        m2.set_xticklabels(ics)
        m2.set_yticklabels(igs)

        plt.tight_layout()
        plt.show()

    if sv_plot:
        plt.clf()
        plt.plot(np.log10(gs), sv_c_constant)
        plt.xlabel("$log_{10}$($\\gamma$)")
        plt.ylabel("Number of Support Vectors")
        plt.title("Number of Support Vectors for Varying $\gamma$-Values")
        plt.show()

        plt.clf()
        plt.plot(np.log10(cs), sv_g_constant)
        plt.xlabel("$log_{10}$(c)")
        plt.ylabel("Number of Support Vectors")
        plt.title("Number of Support Vectors for Varying C-Values")
        plt.show()



if __name__ == "__main__":
    data_train = load_data("IA3-train.csv")
    data_dev = load_data("IA3-dev.csv")

    # sentiment = data_train[:,0]
    # sentences = data_train[:,1]
    #
    # pos_data = data_train[sentiment == 1, :]
    # neg_data = data_train[sentiment == 0, :]
    #
    # X_pos, _, words_pos = preprocess(pos_data, data_dev)
    # X_neg, _, words_neg = preprocess(neg_data, data_dev)
    #
    # idf_pos_vectors = X_pos.toarray()
    # idf_neg_vectors = X_neg.toarray()
    #
    # summed_idfs_pos = np.sum(idf_pos_vectors, axis=0)
    # summed_idfs_neg = np.sum(idf_neg_vectors, axis=0)
    #
    # idxs_pos = np.argpartition(summed_idfs_pos, -10)[-10:]
    # idxs_neg = np.argpartition(summed_idfs_neg, -10)[-10:]
    #
    # print(np.sort(summed_idfs_pos[idxs_pos])[::-1])
    # print(words_pos[idxs_pos][np.argsort(summed_idfs_pos[idxs_pos])][::-1])
    #
    # print(np.sort(summed_idfs_neg[idxs_neg])[::-1])
    # print(words_neg[idxs_neg][np.argsort(summed_idfs_neg[idxs_neg])][::-1])
    #
    # X_pos, _, words_pos = preprocess(pos_data, data_dev, method="Vectorizer")
    # X_neg, _, words_neg = preprocess(neg_data, data_dev, method="Vectorizer")
    #
    # idf_pos_vectors = X_pos.toarray()
    # idf_neg_vectors = X_neg.toarray()
    #
    # summed_idfs_pos = np.sum(idf_pos_vectors, axis=0)
    # summed_idfs_neg = np.sum(idf_neg_vectors, axis=0)
    #
    # idxs_pos = np.argpartition(summed_idfs_pos, -10)[-10:]
    # idxs_neg = np.argpartition(summed_idfs_neg, -10)[-10:]
    #
    # print(np.sort(summed_idfs_pos[idxs_pos])[::-1])
    # print(words_pos[idxs_pos][np.argsort(summed_idfs_pos[idxs_pos])][::-1])
    #
    # print(np.sort(summed_idfs_neg[idxs_neg])[::-1])
    # print(words_neg[idxs_neg][np.argsort(summed_idfs_neg[idxs_neg])][::-1])

    X_train, X_dev, words_train = preprocess(data_train, data_dev)
    
    y_train = data_train[:,0].astype(int)
    y_dev = data_dev[:,0].astype(int)

    ic_base = np.arange(-4, 5, 1)
    ic_fine = np.arange(-1, 1.1, 0.1)

    ig_base = np.arange(-5, 2, 1)
    ig_fine = None

    lin_accuracies = linearSVM_compare(X_train, y_train, X_dev, y_dev, ic_base)
    # quad_accuracies = quadraticSVM_compare(X_train, y_train, X_dev, y_dev, ic_base)
    # rbf_accuracies = rbfSVM_compare(X_train, y_train, X_dev, y_dev, ic_base, ig_base)


     