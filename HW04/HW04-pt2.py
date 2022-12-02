#!/usr/bin/env python3
import re
import sklearn as sk
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from GloVe_Embedder import GloVe_Embedder



# meaningful_word_list.extend([str(i) for i in range(1000)])

def preprocess(data_train, data_dev, method="GloVe 1"):

    if method == "TF-IDF":
        vectorizer = TfidfVectorizer(use_idf=True)
        X_train = vectorizer.fit_transform(data_train[:, 1])
        X_dev = vectorizer.transform(data_dev[:, 1])
        names = vectorizer.get_feature_names_out()

    elif method == "Vectorizer":
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(data_train[:, 1])
        X_dev = vectorizer.transform(data_dev[:, 1])
        words = vectorizer.get_feature_names_out()

    # Just find the average embedding for all the words
    elif method == "GloVe 1":
        ge = GloVe_Embedder("GloVe_Embedder_data.txt")
        tweet_list_train = []
        tweet_list_dev = []
        for tweet in data_train:
            tweet_list_train.append(re.findall(r'[\w]+', tweet[1].lower()))
        for tweet in data_dev:
            tweet_list_dev.append(re.findall(r'[\w]+', tweet[1].lower()))

        X_train = np.zeros((9000, 200))
        for i, tweet in enumerate(tweet_list_train):
            embeddings = ge.embed_list(tweet)
            X_train[i, :] = np.average(embeddings, axis=0)

        X_dev = np.zeros((2500, 200))
        for i, tweet in enumerate(tweet_list_dev):
            embeddings = ge.embed_list(tweet)
            X_dev[i, :] = np.average(embeddings, axis=0)

    # Weighted average for each embedding (although I'm not sure the weighting is great)
    elif method == "GloVe 2":
        ge = GloVe_Embedder("GloVe_Embedder_data.txt")

        vectorizer = TfidfVectorizer(use_idf=True)
        X_train2 = vectorizer.fit_transform(data_train[:, 1])
        X_dev2 = vectorizer.transform(data_dev[:, 1])
        words = vectorizer.get_feature_names_out()

        X_train = np.zeros((9000, 200))
        X_dev = np.zeros((2500, 200))
        for i, tweet in enumerate(X_train2):
            # get index
            feature_index = tweet.nonzero()[1]
            tweet_words = words[feature_index]
            tfidf_scores = [1/tweet[0, x] + 0.0000000001 for x in feature_index]
            embeddings = ge.embed_list(tweet_words)
            embeddings_weighted = np.array([np.array(tfidf_scores) * embeddings[:, i] for i in range(200)])
            X_train[i, :] = np.average(embeddings_weighted, axis=1)

        for i, tweet in enumerate(X_dev2):
            # get index
            feature_index = tweet.nonzero()[1]
            tweet_words = words[feature_index]
            tfidf_scores = [1/tweet[0, x] + 0.0000000001 for x in feature_index]
            embeddings = ge.embed_list(tweet_words)
            embeddings_weighted = np.array([np.array(tfidf_scores) * embeddings[:, i] for i in range(200)])
            X_dev[i, :] = np.average(embeddings_weighted, axis=1)

    # Just find the average embedding for all the words
    elif method == "GloVe 3":
        ge = GloVe_Embedder("GloVe_Embedder_data.txt")

        meaningful_word_list = []
        for word in ge.embedding_dict:
            embedding = ge.embed_str(word, indicate_unk=True, warn_unk=False)
            if not embedding[1]:
                meaningful_word_list.append(word)
        meaningful_word_list.extend([str(i) for i in range(1000)])


        tweet_list_train = []
        tweet_list_dev = []
        for tweet in data_train:
            tweet_list_train.append(re.findall(r'[\w]+', tweet[1].lower()))
        for tweet in data_dev:
            tweet_list_dev.append(re.findall(r'[\w]+', tweet[1].lower()))

        X_train = np.zeros((9000, 200))
        for i, tweet in enumerate(tweet_list_train):
            embeddings = np.zeros((1, 200))
            for p, word in enumerate(tweet):
                embedding = ge.embed_str(word, indicate_unk=True)
                if not embedding[1]:
                    embeddings = np.concatenate((embeddings, np.reshape(embedding[0], (1, -1))), axis=0)
                else:
                    new_words = split_words(word, meaningful_word_list, ge)
                    for new_word in new_words:
                        embedding = ge.embed_str(new_word, indicate_unk=True)
                        if not embedding[1]:
                            embeddings = np.concatenate((embeddings, np.reshape(embedding[0], (1, -1))), axis=0)
            np.delete(embeddings, 0, axis=0)
            X_train[i, :] = np.average(embeddings, axis=0)

        X_dev = np.zeros((2500, 200))
        for i, tweet in enumerate(tweet_list_dev):
            embeddings = np.zeros((1, 200))
            for p, word in enumerate(tweet):
                embedding = ge.embed_str(word, indicate_unk=True)
                if not embedding[1]:
                    embeddings = np.concatenate((embeddings, np.reshape(embedding[0], (1, -1))), axis=0)
                else:
                    new_words = split_words(word, meaningful_word_list, ge)
                    for new_word in new_words:
                        embedding = ge.embed_str(new_word, indicate_unk=True)
                        if not embedding[1]:
                            embeddings = np.concatenate((embeddings, np.reshape(embedding[0], (1, -1))), axis=0)
            np.delete(embeddings, 0, axis=0)
            X_dev[i, :] = np.average(embeddings, axis=0)

    return X_train, X_dev

def parse(data, meaningful_word_list, result=None):
    if result is None:
        result = []
    if data in meaningful_word_list:
        result.append(data)
        yield result[::-1]
    else:
        for i in range(1, len(data)):
            first, last = data[:i], data[i:]
            if last in meaningful_word_list:
                yield from parse(first, result + [last])

def split_words(word, meaningful_word_list, ge):
    word_options = []
    for words in parse(word, meaningful_word_list):
        word_options.append(words)

    if len(word_options) > 0:
        avg = np.zeros(len(word_options))
        for k, words in enumerate(word_options):

            embeddings = ge.embed_list(words)
            dist = np.zeros((len(words), len(words)))
            for i, embed1 in enumerate(embeddings):
                for j, embed2 in enumerate(embeddings):
                    dist[i, j] = np.dot(embed1, embed2)
            avg[k] = np.mean(dist)
        index = np.argmin(avg)
        return word_options[index]
    else:
        return word


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
    cs = np.power(10 * np.ones(i_s.shape), i_s)
    accuracy_test = []
    accuracy_train = []
    num_sv = []
    for i, c in enumerate(cs):
        if c==1:
            print('hi')
        linSVM = SVC(C=c, kernel='linear')
        linSVM.fit(X_train, y_train)
        predictions_test = linSVM.predict(X_dev)
        predictions_train = linSVM.predict(X_train)
        accuracy_test.append(np.sum(predictions_test == y_dev) / np.size(y_dev))
        accuracy_train.append(np.sum(predictions_train == y_train) / np.size(y_train))
        num_sv.append(linSVM.support_.size)
        print("i = {0}, c = {1}, accuracy = {2}%".format(i_s[i], np.round(c, 2), np.round(100 * accuracy_test[i], 3)))

    best_idx = np.argmax(accuracy_test)
    print(num_sv)
    print("----- BEST -----")
    print("i = {0}, c = {1}, accuracy = {2}%".format(i_s[best_idx], np.round(cs[best_idx], 2),
                                                     np.round(100 * accuracy_test[best_idx], 3)))

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


def rbfSVM_compare(X_train, y_train, X_dev, y_dev, ics, igs, heatmap=True, sv_plot=False):
    cs = np.power(10 * np.ones(ics.shape), ics)
    gs = np.power(10 * np.ones(igs.shape), igs)
    # cs = np.array([6, 6])
    # gs = np.array(([0.13, 0.13]))

    accuracy_train = np.zeros((gs.size, cs.size))
    accuracy_dev = accuracy_train.copy()
    sv_c_constant = []
    sv_g_constant = []

    for col, c in enumerate(cs):  # columns
        for row, g in enumerate(gs):  # rows
            rbfSVM = SVC(C=c, kernel='rbf', gamma=g)
            rbfSVM.fit(X_train, y_train)

            predictions_train = rbfSVM.predict(X_train)
            predictions_dev = rbfSVM.predict(X_dev)

            accuracy_train[row, col] = (np.sum(predictions_train == y_train) / np.size(y_train))
            accuracy_dev[row, col] = (np.sum(predictions_dev == y_dev) / np.size(y_dev))
            print("c = {0}, g = {1}, accuracy = {2}%".format(np.round(c, 2), np.round(g, 2),
                                                             np.round(100 * accuracy_dev[row, col], 3)))
            if np.isclose(c, 10):
                sv_c_constant.append(rbfSVM.support_.size)
            if np.isclose(g, 0.1):
                sv_g_constant.append(rbfSVM.support_.size)

    br, bc = np.unravel_index(accuracy_dev.argmax(), accuracy_dev.shape)
    print("------- BEST -------")
    print("c = {0}, g = {1}, accuracy = {2}%".format(np.round(cs[bc], 2), np.round(gs[br], 2),
                                                     np.round(100 * accuracy_dev[br, bc], 3)))

    if heatmap:
        f, (ax1, ax2, cax) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.1], 'height_ratios': [1]})
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


    X_train, X_dev = preprocess(data_train, data_dev)

    y_train = data_train[:, 0].astype(int)
    y_dev = data_dev[:, 0].astype(int)

    # ic_base = np.arange(-4, 5, 1)
    ic_base = np.arange(-1, 3, 1)
    ic_fine = np.arange(-1, 1.1, 0.1)

    ig_base = np.arange(-2, 1, 1)
    ig_fine = None

    lin_accuracies = linearSVM_compare(X_train, y_train, X_dev, y_dev, ic_base)
    # rbf_accuracies = rbfSVM_compare(X_train, y_train, X_dev, y_dev, ic_base, ig_base)