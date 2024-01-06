import numpy as np
import matplotlib.pyplot as plt

def classification_stats(preds, labels, binary=True):
    assert preds.shape == labels.shape, "Predictions and Labels aren't the same length"
    if binary:
        tp = preds[np.logical_and(preds == labels, preds==1)].shape[0]
        fp = preds[np.logical_and(preds != labels, preds==1)].shape[0]
        tn = preds[np.logical_and(preds == labels, preds==0)].shape[0]
        fn = preds[np.logical_and(preds != labels, preds==0)].shape[0]

        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) 
        recall = tp / (tp + fn)
        sens = recall
        return (tp, fp, tn, fn), {"acc": acc, "precision": precision, "recall": recall}
    else:
        acc = preds[preds==labels].shape[0] / preds.shape[0]
        return  {"acc": acc}

def confusion_matrix(tp, fp, tn, fn):
    confusion_matrix = np.array([tn, fp, fn, tp], dtype=int).reshape(2, 2)
    fig, ax = plt.subplots()
    ax.matshow(confusion_matrix, cmap="winter_r")
    ax.invert_yaxis()
    ax.invert_xaxis()
    for (i, j), val in np.ndenumerate(confusion_matrix):
        plt.text(i, j, '{}'.format(val), ha='center', va='center')
        plt.xlabel("Ground Truth")
        plt.ylabel("Predicted")
    plt.show()