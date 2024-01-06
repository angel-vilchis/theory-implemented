import numpy as np
import matplotlib.pyplot as plt 

def scatter_jitter(x1, x2, c, jitter=0.1):
    def rand_jitter(arr):
        stdev = jitter * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    plt.scatter(rand_jitter(x1), rand_jitter(x2), c=c)