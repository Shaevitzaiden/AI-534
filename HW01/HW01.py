#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    data_pd = pd.read_csv(path)
    data_np = data_pd.to_numpy()
    
    x = data_np[:,1:-1]
    print(x.shape)
    y = data_np[:,-1]
    
    # Split dates
    dates_np = np.zeros((data_np.shape[0],3), dtype=np.int)
    for i, row in enumerate(dates_np[:]):
        row[:] = [int(d) for d in x[i,0].split('/')]
    
    # Tack dates to the end
    x = np.hstack((x,dates_np))
    
    # replace old date location with weights
    x[:,1] = 1

    # Year built idx = 12, year renovated idx = 13
    # 
    year = 2022
    for i, row in enumerate(x[:,12:14]):
        if row[1] != 0:
            row[1] = year - row[1]
        else:
            row[1] = year - row[0]
    return x, y


if __name__ == "__main__":
    X, Y = load_data("HW01\IA1_train.csv")
    print(X[:,13])