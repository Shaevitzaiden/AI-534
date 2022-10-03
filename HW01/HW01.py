#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        y = y.reshape((y.shape[0],1))
        
        # Initialize weights as 1 (could change to random in the future)
        weights = np.ones((num_features,1)) # create vertical vector of weights

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
        y = y.reshape((y.shape[0],1))
        return self.get_mse(self.weights, x, y)

    @staticmethod
    def get_mse(w, x, y):
        mse = np.square(np.dot(x, w) - y).mean()
        # print(mse)
        return mse


def load_data(path, normalize=True, sqrt_living15=True):
    data_pd = pd.read_csv(path)
    data_np = data_pd.to_numpy()
    
    x = data_np[:,1:-1]
    y = data_np[:,-1]
    
    # Split dates
    dates_np = np.zeros((data_np.shape[0],3), dtype=np.float64)
    for i, row in enumerate(dates_np[:]):
        row[:] = [int(d) for d in x[i,0].split('/')]

    
    x = np.delete(x, 0, axis=1)
    x = np.hstack((dates_np, x))

    # Year built idx = 15, year renovated idx = 16
    year = 2022
    for i, row in enumerate(x[:,15:17]):
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

    if not sqrt_living15:
        x = np.delete(x, 17, axis=1)
    
    return x, y


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
    


if __name__ == "__main__":
    t1 = time()
    X, Y = load_data("HW01\IA1_train.csv")
    X_dev, Y_dev = load_data("HW01\IA1_dev.csv")
    print("load time: {}".format(time()-t1))
    lrs = [10**-1, 10**-2, 10**-3, 10**-4]
    
    house_models = [LRM() for i in range(len(lrs))] # create a model for each lr
    t1 = time()
    loss_ot = [house_models[i].train(X, Y, 4000, lrs[i]) for i in range(len(house_models))] 
    print("train time = {}".format(time()-t1))
    
    # Get feature weights for each model
    feature_weights = [model.weights for model in house_models]
    print(feature_weights)

    # Get final mse for each model in tuples, MSE = (MSE_train, MSE_dev)
    eval_mse = [(model.mse_ot[-1], model.eval(X_dev, Y_dev)) for model in house_models]
    print(eval_mse)
    
    # Plot of mse over time for each model/learning rate
    plot_mse(loss_ot, lrs)