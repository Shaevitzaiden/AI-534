#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from time import time


if __name__ == "__main__":
    # a = np.random.randint(1, 6, (10,))
    # print(a[1:])
    # m = a[1:] > 3
    # print(m)
    # print(m * a[1:])
    # # maxed_mask = (np.abs(weights[1:]) - lr*reg_param) > 0
    # # maxed = maxed_mask * weights[1:]
    # # weIghts[1:] = np.sign(weights[1:])
    
    # a = np.array([[1, -1, -1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, 1, 1, 1]])
    # b = np.array([12.67, 10.55, 10.04, 8.96])
    # c = 1/2*np.dot(a.T,b)
    # print(c)
    # m = 1/3 * (np.sum(c[1:]))
    # print(m)
    # v = 1/2 * (np.sum((c[1:]-m)**2))
    # print(v)
    # s = np.sqrt((v**2)/2)
    # print(s)

    s1 = np.array([12.33, 10.52, 10.33, 9])
    s2 = np.array([13, 10.57, 9.75, 8.92])

    v = np.sum(((s1-s2)**2)/8)
    print(v)
    print(np.sqrt(v/2))

    # ------------- 3.11 -----------------------
    a = np.array([[1, -1, -1, -1, 1, 1, 1, -1], 
                [1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, 1, -1, 1, -1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, -1, 1, -1, 1, -1, -1],
                [1, -1, 1, 1, -1, -1, 1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1]])

    b = np.array([4.56, 5.98, 13.98, 20.34, 2.01, 3.27, 12.23, 18.69])
    c = 1/2*np.dot(a.T,b)
    c[0] = c[0]/4
    c[1:] = c[1:] / 2
    # print(c)

    s1 = np.array([4.39, 6.21, 14.51, 19.57, 2.09, 3.15, 11.77, 19.4])
    s2 = np.array([4.73, 5.75, 13.45, 21.11, 1.93, 3.39, 12.69, 17.98])

    v = np.sum(((s1-s2)**2)/16)
    print(v)
    print(np.sqrt(v/4))
