import numpy as np

def logistic(x):
    return 1 / (1 + np.exp(-x))

def r_squared(y_pred, y_label):
    return 1 - np.sum((y_pred-y_label)**2) / np.sum((y_pred-np.mean(y_pred))**2)

def logloss(pred, label):
    return -(label*np.log(pred) + (1-label)*np.log(1-pred))

def mode(arr):
    uniques, counts = np.unique(arr, return_counts=True)
    modes = uniques[np.argwhere(counts == np.amax(counts))].flatten()
    return np.random.choice(modes)