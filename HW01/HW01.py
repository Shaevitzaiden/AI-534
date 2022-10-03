#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LRM():
    def __init__(self) -> None:
        self.best_weights = None
        self.mse_ot = None
        self.mse_final = None

    def train(self, x, y, iterations, lr, method="batch") -> np.array:
        # train the model:
        # 1. 
        pass

    def evaluate(self, x, y) -> float:
        # returns the error using mean squared error
        pass

    # @staticmethod
    # def get_mse()


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
    return x, y


if __name__ == "__main__":
    X, Y = load_data("HW01\IA1_train.csv")
    lrs = [10**0, 10**1, 10**2, 10**3, 10**4]
    print(X.shape)