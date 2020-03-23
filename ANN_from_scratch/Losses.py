import numpy as np


def MSE(Y, pred, derivative=False):
    delta = (pred - Y)
    if not derivative:
        print(np.sum(delta ** 2) * 0.5)
        return
    return delta
