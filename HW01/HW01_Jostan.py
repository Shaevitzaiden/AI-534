#!/usr/bin/env python3
# AI 534
# AI1 skeleton code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    loaded_data = pd.read_csv(path)
    return loaded_data

# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize=True, drop_sqrt_living15=False):
    # Your code here:
    # Drop id column
    preprocessed_data = data.drop('id', axis=1)
    # Drop sqft_lifing_15 if needed
    if drop_sqrt_living15:
        preprocessed_data = preprocessed_data.drop('sqft_living15', axis=1)

    # Make date place holder
    date_place_holder = np.empty([preprocessed_data.shape[0], 3], dtype=int)
    for count, dates in enumerate(preprocessed_data["date"].str.rsplit("/")):
        date_place_holder[count, :] = dates

    # Add parsed date to dataframe
    preprocessed_data['day'] = date_place_holder[:, 1]
    preprocessed_data['month'] = date_place_holder[:, 0]
    preprocessed_data['year'] = date_place_holder[:, 2]
    preprocessed_data = preprocessed_data.drop('date', axis=1)

    # Add dummy variable to dataframe
    preprocessed_data['bias'] = 1

    # Calculate the years between when the house was renovated and when it was sold
    age_since_renovated_list = []
    for index in preprocessed_data.index:
        yr_renovated = preprocessed_data['yr_renovated'][index]
        yr_built = preprocessed_data['yr_built'][index]
        year = preprocessed_data['year'][index]
        if yr_renovated == 0:
            age_since_renovated = year - yr_built
        else:
            age_since_renovated = year - yr_renovated
        age_since_renovated_list.append(age_since_renovated)
    # Replace renovation year with years since renovation
    preprocessed_data['age_since_renovated'] = age_since_renovated_list
    preprocessed_data = preprocessed_data.drop('yr_renovated', axis=1)

    # Normalize the columns with z-score method if needed
    if normalize:
        for (columnName, columnData) in preprocessed_data.iteritems():
            if columnName not in ['price', 'waterfront', 'bias']:
                preprocessed_data[columnName] = (preprocessed_data[columnName] - preprocessed_data[columnName].mean()) / preprocessed_data[columnName].std()

    return preprocessed_data

# # Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# # Expand the arguments of this function however you like to control which feature modification
# # approaches are / aren't active.
# def modify_features(data):
#     # Your code here:
#
#     return modified_data

# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, labels, lr, iterations_max=4000, stop_if_converged=False, convergence_criteria=0.0001):
    """
    A function to use gradient decent to train a model.
    :param data: Data in the form of a numpy array
    :param labels:
    :param lr: Learning rate to use
    :param iterations_max: Max number of iterrations to use for gradient decent
    :param stop_if_converged: Boolean to have model stop iterating if it reaches the convergence criteria
    :param convergence_criteria: Criteria to use to know when to stop iterating, if the change in MSE between 2
    iterations is smaller than this, then it will stop.
    :return: Weights for the features of the model as a numpy array and a list of the MSE values for each iterations
    """

    # Your code here:
    num_samples = data.shape[0]

    num_features = data.shape[1]

    # Initialize weights as 1
    weights = np.ones((num_features,))
    mse_ot = []
    mse_ot.append(get_mse(weights, data, labels))

    # train the model:
    # [[x00 x10 x20 x30 ..]   === [y_hat1, y_hat2]
    # [x01 ]]
    for i in range(iterations_max):
        # Compute error (e = y_hat - y)
        error = np.dot(data, weights) - labels  # [e1 e2 e3 ... ]

        # Compute gradient for weights
        grad_Lw = 2 / num_samples * np.dot(data.T, error)

        # Update weights
        weights = weights - lr * grad_Lw

        # Record MSE
        mse_ot.append(get_mse(weights, data, labels))

        # Stop iterating if the convergence criteria is met
        if mse_ot[i]-mse_ot[i+1] < convergence_criteria and stop_if_converged:
            break
        # Update weights

    return weights, mse_ot

def get_mse(w, x, y):
    mse = np.square(np.dot(x, w) - y).mean()
    return mse


# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses, lrs=[0.1, 0.01, 0.001, 0.0001]):
    fig, ax = plt.subplots()
    # plot lines and add legend labels to them
    lines = []
    for i, lr in enumerate(lrs):
        try:
            lines.append(ax.plot(losses[:100,i], label=str(lr)))
        except IndexError:
            lines.append(ax.plot(losses, label=str(lr)))
    ax.legend(title="lrs", fontsize="small", fancybox=True, loc='upper right')
    ax.set_ylabel("MSE")
    ax.set_xlabel("Iteration")
    ax.set_title("MSE vs Training Iterations")
    # plt.ylim([0, 50])
    plt.show()
    return

# Invoke the above functions to implement the required functionality for each part of the assignment.
# Part 0  : Data preprocessing.
# Your code here:
# Load the training and test data as dataframes
training_set = load_data('IA1_train.csv')
test_set = load_data('IA1_dev.csv')

# Part 1 . Implement batch gradient descent and experiment with different learning rates.
# Your code here:
# Set training rates and number of iterations
training_rates = [0.1, 0.01, 0.001, 0.0001]
max_iterations = 4000
# Modify training data
data_train = preprocess_data(training_set)
# Pull out the price variable as the prediction objective
labels = data_train['price'].to_numpy()
data_train = data_train.drop('price', axis=1)
# Convert the data to a numpy array
data_train = data_train.to_numpy()

# Modify testing data
data_test = preprocess_data(test_set)
# Pull out prediction objective
labels_test = data_test['price'].to_numpy()
data_test = data_test.drop('price', axis=1)
# Convert testing data to a numpy array
data_test_np = data_test.to_numpy()

# Setup the arrays for weights and MSE
weights = np.zeros([len(training_rates), data_train.shape[1]])
losses = np.empty([max_iterations+1, len(training_rates)])
# Set MSE array to nan so that it can plot even if not full
losses[:] = np.nan

# Train models using each of the training rates and record the weights and MSEs for each model
for i, training_rate in enumerate(training_rates):
    weights[i, :], losses_temp = gd_train(data_train, labels, training_rate, stop_if_converged=True)
    # print(len(losses_temp))
    losses[:len(losses_temp), i] = losses_temp
    # print(get_mse(weights[i, :], data_test_np, labels_test))
 # plot_losses(losses)


# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:
# Modify training data, this time don't normalize the data
data_train = preprocess_data(training_set, normalize=False)
# Pull out the price variable as the prediction objective
labels = data_train['price'].to_numpy()
data_train = data_train.drop('price', axis=1)
# Convert the data to a numpy array
data_train_np = data_train.to_numpy()
# Train the model
weights, losses = gd_train(data_train_np, labels, 10**-11)
# plot_losses(np.array(losses), lrs=[10**-11])

# Modify testing data
data_test = preprocess_data(test_set, normalize=False)
# Pull out prediction objective
labels_test = data_test['price'].to_numpy()
data_test = data_test.drop('price', axis=1)
# Convert testing data to a numpy array
data_test = data_test.to_numpy()
# print(get_mse(weights, data_train, labels))
# print(get_mse(weights, data_test, labels_test))
# print(weights)

# Part 2 b Training with redundant feature removed.
# Your code here:
# Modify training data, this time remove the square footage of nearby houses feature
data_train = preprocess_data(training_set, drop_sqrt_living15=True)
# Pull out the price variable as the prediction objective
labels = data_train['price'].to_numpy()
data_train = data_train.drop('price', axis=1)
# Convert the data to a numpy array
data_train = data_train.to_numpy()
# Train the model
weights, losses = gd_train(data_train, labels, 0.1, stop_if_converged=True)
# plot_losses(np.array(losses), lrs=[10**-1])

# Modify testing data
data_test = preprocess_data(test_set, drop_sqrt_living15=True)
# Pull out prediction objective
labels_val = data_test['price'].to_numpy()
data_test = data_test.drop('price', axis=1)
# Convert testing data to a numpy array
data_test = data_test.to_numpy()
# print(get_mse(weights, data_test, labels_val))
# print(weights)


