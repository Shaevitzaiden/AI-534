#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


class LRM():
    def __init__(self) -> None:
        # Initialize weights and variables to save
        self.weights = None
        self.best_weights = None
        self.mse_ot = None
        self.num_iterations = None

    def train(self, x, y, converge_thresh, max_iterations, lr, reg_param, method="l2") -> np.array:
        """
        Trains a logistic regression model
        :param x: Feature values of examples
        :param y: Example response
        :param converge_thresh: Convergence threshold, this is the change in MSE that the model will stop iterating at
        :param max_iterations: Max number of iterations the model trains for
        :param lr: Learning rate
        :param reg_param: Regularization Parameter
        :param method: Type of regularization to use
        :return: Array of MSE values for each iteration
        """

        num_samples = x.shape[0]
        num_features = x.shape[1]

        # Initialize weights as 0
        weights = np.zeros((num_features,))

        # Create list to save MSE values for each iteration
        mse_ot = []
        mse_ot.append(self.get_mse(weights, x, y))


        # Initialize delta error variable and variable to save MSE value for checking the change in the MSE
        delta_error = np.Inf
        mse_prev = np.Inf
        iterations = 0
        # Train model
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

            # Record MSE values
            mse = self.get_mse(weights, x, y)
            mse_ot.append(mse)

            # Record the change in the MSE for checking convergence
            delta_error = mse_prev - mse
            mse_prev = mse
            iterations += 1

        # Record model weights, MSE values for each iteration, and the number of iterations it took to train the model
        self.weights = weights
        self.mse_ot = mse_ot
        self.num_iterations = iterations
        return mse_ot

    def eval(self, x, y) -> float:
        """
        Finds the accuracy of the model
        :param x: weights of testing examples
        :param y: correct response
        :return: accuracy of model (0 - 1)
        """
        prediction = self.predict(x)
        prediction[prediction > 0.5] = 1
        prediction[prediction <= 0.5] = 0
        return np.sum(prediction == y)/len(y)

    def predict(self, x) -> np.array:
        """
        Calculate estimate using feature values and model weights
        :param x: Feature values
        :return: Response estimate
        """
        return self.sigmoid(
            np.dot(x, self.weights))

    def get_mse(self, w, x, y) -> float:
        """
        Calcuate the MSE of the model
        :param w: model weights
        :param x: feature values
        :param y: Correct response
        :return: MSE value of model
        """
        mse = np.square(self.sigmoid(np.dot(x, w)) - y).mean()
        return mse

    @staticmethod
    def sigmoid(x) -> np.array:
        return 1 / (1 + np.exp(-x))


def load_data(path, normalize=True, remove_col=None, test=False, test_data=False, stds=None, means=None):
    """
    Loads and pre-processes data
    :param path: Path to data
    :param normalize: Boolean to toggle normalization
    :param remove_col: Columns to remove from data
    :param test: Boolean to set whether correct response is included in data
    :param test_data: Boolean to set whether this is training or testing data
    :param stds: Standard deviations to use for normalization if it's testing data
    :param means: Mean to use for normalization if it's testing data
    :return: processed data, column headers, stds and mean if it's training data
    """
    # Load data and seperate data into numpy array and headers
    data_pd = pd.read_csv(path)
    data_np = data_pd.to_numpy()
    column_headers = np.asarray(data_pd.columns)

    if test:
        x = data_np[:, :]
        y = None
    else:
        x = data_np[:, :-1]
        y = data_np[:, -1]

    # Columns of data to normalize
    norm_idx = [2, 6, 7]

    # If it is not test data, but needs normalization, then normalize and save the stds and means for use with training
    # data
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
    # If it's training data, then use the input stds and means for normalization
    elif normalize:
        x = x.astype('float64')
        for i, mean, std in zip(norm_idx, means, stds):
            x[:, i] = (x[:, i] - mean) / std

    # Remove columns if needed
    if remove_col is not None:
        x = np.delete(x, remove_col, axis=1)

    # If it training data, return the stds, means, and columns headers as well as the data.
    if test_data:
        return x, y
    else:
        return x, y, column_headers, stds, means


def plot_mse(mse_arrays, lrs, reg_params):
    """
    Plot the mse values
    :param mse_arrays: List of MSE arrays, each array having MSE for each iteration of model training
    :param lrs: List of learning rates
    :param reg_params: List of regularization parameters
    :return: Plots MSE
    """
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
    """
    Plots the accuracy vs the log of the regularization parameter
    :param train: Training accuracies
    :param test: Testing accuracies
    :param reg_params: List of regularization parameters corresponding to accuracies
    :return: Plot of accuracy
    """
    fig, ax = plt.subplots()
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
    """
    Plots the sparsity vs the log of the coresponding regularization parameters
    :param sparsity: Sparsity of each model
    :param reg_params: Coresponding regularization parameters
    :return: Plot of sparsity
    """
    fig, ax = plt.subplots()
    ax.plot(reg_params, sparsity, label='train')
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
    """
    Function to do analysis for each part of the assignments
    :param train_path: Path for training data
    :param test_path: Path for testing data
    :param lrs: Learning rates to use
    :param method: Regularization method to use
    :param plot_accuracy_toggle: Boolean to toggle accuracy plot on/off
    :param plot_sparsity_toggle: Boolean to toggle sparsity plot on/off
    :param print_weights: Boolean to toggle printing of top weights
    :param plot_MSE_toggle: Boolean to toggle MSE plot on/off
    :param print_results: Boolean to toggle printing of results
    :return:
    """

    # Pre-process training data and record stds and means
    X, Y, column_headers, stds, means = load_data(train_path)
    # Pre-process testing data, using normalization parameters from training data
    X_test, Y_test = load_data(test_path, means=means, stds=stds, test_data=True)

    # Array of regularization parameters
    reg_params = 10 ** np.arange(-6, 3, dtype=float)
    # Convergence threshold
    converge_thresh = 10e-8
    # Max number of iterations to train model for
    max_iterations = 3000

    # Create model objects to train
    models = [LRM() for i in range(len(reg_params))]

    # Setup lists to store model results
    train_results = []
    test_results = []
    sparsity = []

    # Train each model on the corresponding regularization parameters and learning rates
    for i in range(len(reg_params)):
        # Train model
        models[i].train(X, Y, converge_thresh, max_iterations, lrs[i], reg_params[i], method)

        # Get model weights
        weights = models[i].weights[1:]
        # Get index of top 5 weights
        top5_weights_index = np.argsort(abs(weights))[-5:]

        # Get accuracy of training and testing data
        train_results.append(models[i].eval(X, Y))
        test_results.append(models[i].eval(X_test, Y_test))

        # Print the results and/or weights, depending on toggles
        if print_results:
            print(
                "Train: " + str(round(train_results[i], 4)) + " Test: " + str(round(test_results[i], 4)) + " iters: " + str(
                    models[i].num_iterations))
        if print_weights:
            for j in np.flip(top5_weights_index):
                print(column_headers[j+1] + ": " + str(round(weights[j], 4)))
        # Save the sparsity value
        sparsity.append(np.sum(np.absolute(weights) <= 10**-6))

    # Plot things if toggles on
    if plot_sparsity_toggle:
        plot_sparsity(sparsity, reg_params)
    if plot_MSE_toggle:
        mses = [np.array(models[i].mse_ot) for i in range(len(models))]
        plot_mse(mses, lrs, reg_params)
    if plot_accuracy_toggle:
        plot_accuracy(train_results, test_results, reg_params)


if __name__ == "__main__":

    # Chosen learning rates for each part of assignment, chosen to minimise the error
    lrs = [10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-2, 10e-2, 10e-3, 10e-4]
    lrs_noisy = [10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-2, 10e-2, 10e-3, 10e-4]
    lrs_l1 = [10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-1, 10e-1]

    # Part 1 analysis
    do_analysis("IA2-train.csv", "IA2-dev.csv", lrs, method='l2', plot_accuracy_toggle=False,
                plot_sparsity_toggle=False, print_weights=False, plot_MSE_toggle=False, print_results=True)
    # Part 2 analysis
    do_analysis("IA2-train-noisy.csv", "IA2-dev.csv", lrs_noisy, method='l2', plot_accuracy_toggle=False,
                plot_sparsity_toggle=False, print_weights=False, plot_MSE_toggle=False, print_results=True)
    # Part 3 analysis
    do_analysis("IA2-train.csv", "IA2-dev.csv", lrs_l1, method='l1', plot_accuracy_toggle=False,
                plot_sparsity_toggle=False, print_weights=False, plot_MSE_toggle=False, print_results=True)

