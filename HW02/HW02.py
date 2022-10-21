#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from time import time


class LRM():
    def __init__(self) -> None:
        self.weights = None
        self.best_weights = None
        self.mse_ot = None
        self.num_iterations = None

    def train(self, x, y, converge_thresh, max_iterations, lr, reg_param, method="l2") -> np.array:
        """
        params: method: str: l1 or l2
        """
        num_samples = x.shape[0]

        # # Add bias column
        # x = np.hstack((x,np.ones((num_samples,1))))

        num_features = x.shape[1]

        # Initialize weights as 1 (could change to random in the future)
        weights = np.zeros((num_features,))  # create vertical vector of weights

        mse_best = np.Inf
        mse_ot = []
        mse_ot.append(self.get_mse(weights, x, y))

        # train the model:
        # [[x00 x10 x20 x30 ..]   === [y_hat1, y_hat2]
        # [x01 ]]
        delta_error = np.Inf
        mse_prev = np.Inf
        iterations = 0
        while (delta_error > converge_thresh) and (iterations < max_iterations):
            # Compute error (e = y_hat - y)
            error = y - self.sigmoid(np.dot(x, weights))  # [e1 e2 e3 ... ]

            # Compute gradient for weights
            grad_Lw = 1 / num_samples * np.dot(x.T, error)

            # Update weights
            weights = weights + lr * grad_Lw
            if method == "l2":
                weights[1:] = weights[1:] - lr * reg_param * weights[1:]
            elif method == "l1":
                reg_vals = np.abs(weights[1:]) - lr * reg_param
                mask = reg_vals > 0
                weights[1:] = np.sign(weights[1:]) * (reg_vals * mask)

            mse = self.get_mse(weights, x, y)
            mse_ot.append(mse)

            delta_error = mse_prev - mse
            mse_prev = mse
            iterations += 1

        self.weights = weights
        self.mse_ot = mse_ot
        self.num_iterations = iterations
        return mse_ot

    def eval(self, x, y) -> float:
        prediction = self.predict(x)
        prediction[prediction > 0.5] = 1
        prediction[prediction <= 0.5] = 0
        return np.sum(prediction == y)

    def predict(self, x) -> np.array:

        # num_samples = x.shape[0]
        # x = np.hstack((x,np.ones((num_samples,1))))
        return self.sigmoid(
            np.dot(x, self.weights))  # ----------------------------- double check if sigmoid should be here

    def loss(w, x, type):
        loss = "poop"
        return loss

    def get_mse(self, w, x, y) -> float:  # ------------------------------ double check if sigmoid should be in here
        mse = np.square(self.sigmoid(np.dot(x, w)) - y).mean()
        return mse

    @staticmethod
    def sigmoid(x) -> np.array:
        return 1 / (1 + np.exp(-x))


def load_data(path, normalize=True, remove_col=None, test=False, test_data=False, stds=None, means=None):
    data_pd = pd.read_csv(path)
    data_np = data_pd.to_numpy()
    column_headers = list(data_pd.columns)
    if test:
        x = data_np[:, :]
        y = None
    else:
        x = data_np[:, :-1]
        y = data_np[:, -1]

    if normalize and not test_data:
        norm_idx = [2, 6, 7]
        x = x.astype('float64')
        means = []
        stds = []

        for i in norm_idx:
            std_x = np.std(x[:, i], axis=0)
            mean_x = np.mean(x[:, i], axis=0)
            x[:, i] = (x[:, i] - mean_x) / std_x
            means.append(mean_x)
            stds.append(std_x)

    elif normalize:
        norm_idx = [2, 6, 7]
        x = x.astype('float64')
        for i, mean, std in zip(norm_idx, means, stds):
            # std_x = np.std(x[:,i], axis=0)
            # mean_x = np.mean(x[:,i], axis=0)
            x[:, i] = (x[:, i] - mean) / std

    if remove_col is not None:
        x = np.delete(x, remove_col, axis=1)

    if test_data:
        return x, y
    else:
        return x, y, column_headers, stds, means


def plot_mse(mse_arrays, lrs, reg_params):
    fig, ax = plt.subplots()

    # plot lines and add legend labels to them
    lines = []
    for mse, lr, reg_param in zip(mse_arrays, lrs, reg_params):
        lines.append(ax.plot(mse, label=str(lr) + ';  ' + str(reg_param)))
    ax.legend(title="LR; RP", fontsize="small", fancybox=True, loc='upper right')
    ax.set_ylabel("MSE")
    ax.set_xlabel("Iteration")
    ax.set_title("MSE vs Training Iterations")
    plt.show()
    return fig, ax


def plot_accuracy(train, test, reg_params):
    fig, ax = plt.subplots()

    # plot lines and add legend labels to them
    lines = []
    ax.plot(reg_params, train, label='train')
    ax.plot(reg_params, test, label='test')
    ax.legend(title="Dataset", fontsize="small", fancybox=True, loc='upper right')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Regularization Parameter")
    ax.set_title("Accuracy vs. Regularization Parameter")
    ax.set_xscale('log')
    plt.show()
    return fig, ax


def save_array_as_csv(file_name, data):
    np.savetxt(file_name, data, delimiter=",")


def write_results_to_csv(file_name, ids, predictions):
    data_to_write = [['id', 'price']]
    for i, p in enumerate(predictions):
        data_to_write.append([str(ids[i]), str(p)])

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_to_write)


if __name__ == "__main__":
    # month=0, day=1, zipcode=16, lat=17, long=18, sq_living15=20

    rm_cols = None  # or None
    X, Y, column_headers, means, stds = load_data("IA2-train.csv")
    column_headers = np.asarray(column_headers)
    X_test, Y_test = load_data("IA2-dev.csv", means=means, stds=stds, test_data=True)

    X_noisy, Y_noisy, _, mean_noisy, std_noisy = load_data("IA2-train-noisy.csv")
    X_test_noisy, Y_test_noisy = load_data("IA2-dev.csv", means=mean_noisy, stds=std_noisy, test_data=True)

    u = 10 ** -2
    reg_params = 10 ** np.arange(-4, 3, dtype=float)
    # lrs = [10e-1, 10e-1, 10e-1, 10e-2, 10e-3, 10e-3, 10e-4]
    lrs = [10e-1, 10e-1, 10e-1, 10e-2, 10e-3, 10e-3, 10e-4]
    converge_thresh = 10e-8
    max_iterations = 3000
    method = 'l2'
    models = [LRM() for i in range(len(reg_params))]
    models_noisy = [LRM() for i in range(len(reg_params))]

    loss_ot = [models[i].train(X, Y, converge_thresh, max_iterations, lrs[i], reg_params[i], method) for i in
               range(len(models))]
    loss_ot_noisy = [
        models_noisy[i].train(X_noisy, Y_noisy, converge_thresh, max_iterations, lrs[i], reg_params[i], method) for i in
        range(len(models))]

    # plot_mse([loss_ot], [10e-4])
    train_results = []
    test_results = []
    top5_weights = []
    top5_weights_index = []

    for i in range(len(reg_params)):
        top5_weights.append(np.sort(abs(models[i].weights[1:]))[-5:])
        top5_weights_index.append(1 + np.argsort(abs(models[i].weights[1:]))[-5:])

        train_results.append(models[i].eval(X, Y) / len(Y))
        test_results.append(models[i].eval(X_test, Y_test) / len(Y_test))
        print(
            "Train: " + str(round(train_results[i], 4)) + " Test: " + str(round(test_results[i], 4)) + " iters: " + str(
                models[i].num_iterations))
        # print(top5_weights[i])
        #
        # print(column_headers[top5_weights_index[i]])

    train_results_noisy = []
    test_results_noisy = []
    for i in range(len(reg_params)):
        # top5_weights.append(np.sort(abs(models[i].weights[1:]))[-5:])
        # top5_weights_index.append(1 + np.argsort(abs(models[i].weights[1:]))[-5:])

        train_results_noisy.append(models_noisy[i].eval(X_noisy, Y_noisy) / len(Y_noisy))
        test_results_noisy.append(models_noisy[i].eval(X_test_noisy, Y_test_noisy) / len(Y_test_noisy))
        print(
            "Train: " + str(round(train_results_noisy[i], 4)) + " Test: " + str(
                round(test_results_noisy[i], 4)) + " iters: " + str(
                models[i].num_iterations))
        # print(top5_weights[i])
        #
        # print(column_headers[top5_weights_index[i]])
    mses = [np.array(models[i].mse_ot) for i in range(len(models))]
    # plot_mse(mses, lrs, reg_params)
    # plot_accuracy(train_results, test_results, reg_params)
    plot_accuracy(train_results_noisy, test_results_noisy, reg_params)


    