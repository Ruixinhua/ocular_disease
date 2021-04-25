import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def accuracy(output, target):
    # calculate accuracy
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    correct = 0
    correct += np.sum(pred == target)
    return round(correct / len(target), 4)


def auc(output, target):
    # calculate ACU score
    return round(roc_auc_score(np.asarray(target), np.asarray(output)[:, 1]), 4)
