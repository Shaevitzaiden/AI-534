#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from time import time


if __name__ == "__main__":
    a = np.random.randint(1, 6, (10,))
    print(a[1:])
    m = a[1:] > 3
    print(m)
    print(m * a[1:])
    # maxed_mask = (np.abs(weights[1:]) - lr*reg_param) > 0
    # maxed = maxed_mask * weights[1:]
    # weIghts[1:] = np.sign(weights[1:])