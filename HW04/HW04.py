#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GloVe_Embedder import *













if __name__ == "__main__":
    embedder = GloVe_Embedder("HW04/GloVe_Embedder_data.txt")
    
    # Part 1 - Build your own data set of words
    seed_words = ["flight", "good", "terrible", "help", "late"]

    num_similar_words = 29

    flight_seed = embedder.find_k_nearest("flight", num_similar_words)
    good_seed = embedder.find_k_nearest("good", num_similar_words)
    terrible_seed = embedder.find_k_nearest("terrible", num_similar_words)
    help_seed = embedder.find_k_nearest("help", num_similar_words)
    late_seed = embedder.find_k_nearest("late", num_similar_words)

    # --- Test --- Trying to put the above into an for loop
    # similar_words = np.zeros((num_similar_words, len(seed_words)))
    # similar_words = np.zeros(len(seed_words))
    # print(similar_words)

    # for i, word in enumerate(seed_words):
    #     similar_words[:, i] = embedder.find_k_nearest(word, num_similar_words)
    # print(similar_words)