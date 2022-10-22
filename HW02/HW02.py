#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


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

        num_features = x.shape[1]

        # Initialize weights as 0
        weights = np.zeros((num_features,))  # create vertical vector of weights

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
    column_headers = np.asarray(data_pd.columns)

    if test:
        x = data_np[:, :]
        y = None
    else:
        x = data_np[:, :-1]
        y = data_np[:, -1]

    norm_idx = [2, 6, 7]

    if normalize and not test_data:
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
        x = x.astype('float64')
        for i, mean, std in zip(norm_idx, means, stds):
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

def plot_sparsity(sparsity, reg_params):
    fig, ax = plt.subplots()

    # plot lines and add legend labels to them
    lines = []
    ax.plot(reg_params, sparsity, label='train')
    # ax.legend(title="Dataset", fontsize="small", fancybox=True, loc='upper right')
    ax.set_ylabel("Sparsity")
    ax.set_xlabel("Regularization Parameter")
    ax.set_title("Sparsity vs. Regularization Parameter")
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

def do_analysis(train_path, test_path, lrs, method, plot_accuracy_toggle=False, plot_sparsity_toggle=False,
                print_weights=False, plot_MSE_toggle=False, print_results=False):

    X, Y, column_headers, stds, means = load_data(train_path)

    X_test, Y_test = load_data(test_path, means=means, stds=stds, test_data=True)

    reg_params = 10 ** np.arange(-6, 3, dtype=float)
    converge_thresh = 10e-8
    max_iterations = 3000

    models = [LRM() for i in range(len(reg_params))]


    train_results = []
    test_results = []
    sparsity = []

    for i in range(len(reg_params)):
        models[i].train(X, Y, converge_thresh, max_iterations, lrs[i], reg_params[i], method)
        weights = models[i].weights[1:]
        top5_weights_index = np.argsort(abs(weights))[-5:]

        train_results.append(models[i].eval(X, Y) / Y.shape[0])
        test_results.append(models[i].eval(X_test, Y_test) / Y_test.shape[0])
        if print_results:
            print(
                "Train: " + str(round(train_results[i], 4)) + " Test: " + str(round(test_results[i], 4)) + " iters: " + str(
                    models[i].num_iterations))
        if print_weights:
            for j in np.flip(top5_weights_index):
                print(column_headers[j+1] + ": " + str(round(weights[j], 4)))

        sparsity.append(np.sum(np.absolute(weights) <= 10**-6))

    if plot_sparsity_toggle:
        plot_sparsity(sparsity, reg_params)
    if plot_MSE_toggle:
        mses = [np.array(models[i].mse_ot) for i in range(len(models))]
        plot_mse(mses, lrs, reg_params)
    if plot_accuracy_toggle:
        plot_accuracy(train_results, test_results, reg_params)


if __name__ == "__main__":

    lrs = [10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-2, 10e-2, 10e-3, 10e-4]
    lrs_noisy = [10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-2, 10e-2, 10e-3, 10e-4]
    lrs_l1 = [10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-1]

    do_analysis("IA2-train.csv", "IA2-dev.csv", lrs, method='l2', plot_accuracy_toggle=True,
                plot_sparsity_toggle=False, print_weights=False, plot_MSE_toggle=False, print_results=True)

    # do_analysis("IA2-train-noisy.csv", "IA2-dev.csv", lrs_noisy, method='l2', plot_accuracy_toggle=False,
    #             plot_sparsity_toggle=False, print_weights=False, plot_MSE_toggle=False, print_results=False)

    # do_analysis("IA2-train.csv", "IA2-dev.csv", lrs_l1, method='l1', plot_accuracy_toggle=True,
    #             plot_sparsity_toggle=True, print_weights=True, plot_MSE_toggle=True, print_results=True)


