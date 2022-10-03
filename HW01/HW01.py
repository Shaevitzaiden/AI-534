#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LRM():
    def __init__(self) -> None:
        self.weights = None
        self.best_weights = None

    def train(self, x, y, iterations, lr, method="batch") -> np.array:
        num_samples = x.shape[0]
        mse_best = np.Inf
        mse_ot = []
        
        # Initialize weights as 1 (could change to random in the fiuture)
        self.weights = np.ones((num_samples,1)) # create vertical vector of weights
        self.best_weights = np.copy(self.best_weights) # make a shallow copy of the weights
        
        # train the model:
        for i in range(iterations):
            # print(self.weights.shape)
            # print(x.shape)
            # print(y.shape)
            # print("--------------------")
            loss_grad = (np.dot(self.weights.T,x) - y)
            loss_grad = 2/num_samples * np.dot(loss_grad,x.T)
            self.weights = self.weights - lr*loss_grad.T
            
            mse_ot.append(self.get_mse(self.weights, x, y))
        return mse_ot

    def evaluate(self, x, y) -> float:
        # returns the error using mean squared error
        pass

    @staticmethod
    def get_mse(w, x, y):
        mse = (np.dot(w.T,x) - y)**2
        mse = np.mean(mse)
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
    
    # Tack dates to the end
    x = np.hstack((x,dates_np))
    
    # replace old date location with weights
    x[:,0] = 1

    # Year built idx = 12, year renovated idx = 13
    year = 2022
    for i, row in enumerate(x[:,12:14]):
        if row[1] != 0:
            row[1] = year - row[1]
        else:
            row[1] = year - row[0]

    if normalize:
        # normalize (minus weights (divide by zero) idx= 0, waterfront idx = 6)
        x = x.astype('float64')
        std_x = np.std(x, axis=0)
        mean_x = np.mean(x, axis=0)
        x[:,1:6] = (x[:,1:6] - mean_x[1:6]) / std_x[1:6]
        x[:,7:] = (x[:,7:] - mean_x[7:]) / std_x[7:]

    if not sqrt_living15:
        x = np.delete(x, 17, axis=1)
    
    # transpose the x_matrix so each sample vector is vertical 
    return x.T, y


def plot_mse(mse_arrays, lrs):
    fig = plt.figure()
    ax = plt.subplot(fig,111)
    fig.add_subplot(ax)

    # plot lines and add legend labels to them
    lines = []
    for mse, lr in zip(mse_arrays, lrs):
        lines.append(ax.plot(mse, label=str(lr)))
    ax.legend(handles=lines, title="lrs", loc=4, fontsize="small", fancybox=True)
    ax.set_ylabel("MSE")
    ax.set_xlabel("Iteration")
    ax.set_title("MSE vs Training Iterations")
    plt.show()
    return fig, ax
    


if __name__ == "__main__":
    X, Y = load_data("HW01\IA1_train.csv")
    lrs = [10**0, 10**-1, 10**-2, 10**-3, 10**-4]
    
    house_model = LRM()
    loss = house_model.train(X, Y, 100, lrs[3])
    plt.plot(loss)
    plt.show()