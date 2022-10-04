#!/usr/bin/env python3


from fileinput import filename
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

    def train(self, x, y, iterations, lr, method="batch") -> np.array:
        num_samples = x.shape[0]
        
        # Add bias column
        x = np.hstack((x,np.ones((num_samples,1))))
        num_features = x.shape[1]
        
        # reshape y into vertical vector
        # y = y.reshape((y.shape[0],1))
        
        # Initialize weights as 1 (could change to random in the future)
        weights = np.ones((num_features,)) # create vertical vector of weights

        mse_best = np.Inf
        mse_ot = []
        mse_ot.append(self.get_mse(weights, x, y))

        # train the model:
        for i in range(iterations):
            # Compute error (e = y_hat - y)
            error = np.dot(x, weights) - y
            
            # Compute gradient for weights
            grad_Lw = 2/num_samples * np.dot(x.T, error)
            
            # Update weights
            weights = weights - lr*grad_Lw
            
            mse_ot.append(self.get_mse(weights, x, y))
        
        self.weights = weights
        self.mse_ot = mse_ot
        return mse_ot

    def eval(self, x, y) -> float:
        # Add bias column to dev data
        num_samples = x.shape[0]
        x = np.hstack((x,np.ones((num_samples,1))))
        # y = y.reshape((y.shape[0],1))
        return self.get_mse(self.weights, x, y)
    
    def predict(self, x) -> np.array:
        num_samples = x.shape[0]
        x = np.hstack((x,np.ones((num_samples,1))))
        return np.dot(x, self.weights)

    @staticmethod
    def get_mse(w, x, y):
        mse = np.square(np.dot(x, w) - y).mean()
        # print(mse)
        return mse


def load_data(path, normalize=True, remove_col=None, test=False):
    data_pd = pd.read_csv(path)
    data_np = data_pd.to_numpy()
    
    if test:
        x = data_np[:,1:]
        y = None
        ids = data_np[:,0]
    else:    
        x = data_np[:,1:-1]
        y = data_np[:,-1]
        ids = data_np[:,0]
    
    # Split dates
    dates_np = np.zeros((data_np.shape[0],3), dtype=np.float64)
    for i, row in enumerate(dates_np[:]):
        row[:] = [int(d) for d in x[i,0].split('/')]

    
    x = np.delete(x, 0, axis=1)
    x = np.hstack((dates_np, x))

    # Year built idx = 15, year renovated idx = 16
    year = 2022
    for i, row in enumerate(x[:,14:16]):
        if row[1] != 0:
            row[1] = year - row[1]
        else:
            row[1] = year - row[0]

    if normalize:
        # normalize (minus waterfront idx = 8)
        x = x.astype('float64')
        std_x = np.std(x, axis=0)
        mean_x = np.mean(x, axis=0)
        x[:,:8] = (x[:,:8] - mean_x[:8]) / std_x[:8]
        x[:,9:] = (x[:,9:] - mean_x[9:]) / std_x[9:]


    if remove_col is not None:
        x = np.delete(x, remove_col, axis=1)
    
    return x, y, ids

def plot_mse(mse_arrays, lrs):
    fig, ax = plt.subplots()

    # plot lines and add legend labels to them
    lines = []
    for mse, lr in zip(mse_arrays, lrs):
        lines.append(ax.plot(mse, label=str(lr)))
    ax.legend(title="lrs", fontsize="small", fancybox=True, loc='upper right')
    ax.set_ylabel("MSE")
    ax.set_xlabel("Iteration")
    ax.set_title("MSE vs Training Iterations")
    plt.show()
    return fig, ax

def save_array_as_csv(file_name, data):
    np.savetxt(file_name, data, delimiter=",")

def write_results_to_csv(file_name, ids, predictions):
    data_to_write = [['id', 'price']]    
    for i, p in enumerate(predictions):
        data_to_write.append([str(ids[i]), str(p)])

    with open (file_name, 'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_to_write)


if __name__ == "__main__":
    # month=0, day=1, zipcode=16, lat=17, long=18, sq_living15=20
    rm_cols = [0, 1, 16]
    X, Y, ids = load_data("HW01\PA1_train1.csv",remove_col=rm_cols)    
    X_test, Y_test, ids_test = load_data("HW01\PA1_test1.csv",remove_col=rm_cols, test=True)
    
    lrs = [10**-1]#, 10**-2, 10**-3, 10**-4]
    
    house_models = [LRM() for i in range(len(lrs))] # create a model for each lr
    loss_ot = [house_models[i].train(X, Y, 4000, lrs[i]) for i in range(len(house_models))] 
    
    # Get feature weights for each model
    feature_weights = [model.weights for model in house_models]
    # print(feature_weights)

    # Get final mse for each model in tuples, MSE = (MSE_train, MSE_dev)
    eval_mse = [model.mse_ot[-1] for model in house_models]
    print(eval_mse)
    
    # Plot of mse over time for each model/learning rate
    # plot_mse(loss_ot, lrs)

    # Save predictions for kaggle
    predictions = house_models[0].predict(X_test)
    print(predictions)
    ids_test = ids_test.astype(np.int64)
    
    write_results_to_csv("HW01\PA1_predictions_0-1-16.csv", ids_test, predictions)




    