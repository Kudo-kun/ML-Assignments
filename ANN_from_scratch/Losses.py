import numpy as np


def MSE(Y, pred):
    delta = (pred - Y)
    error = (np.sum(delta ** 2) * 0.5)
    return (error, delta)


def binary_crossentropy(Y, pred):
    sz = pred.shape[0]
    error, delta = 0, []
    for i in range(sz):
        if Y[i] == 1:
            error += np.log(pred[i])
            delta.append(1 / pred[i])
        elif Y[i] == 0:
            error += (1 - np.log(pred[i]))
            delta.append(1 / (1 - pred[i]))
    delta = np.array(delta)
    return (-error, -delta)


def categorical_crossentropy(Y, pred):
    sz = pred.shape[0]
    error, delta = 0, []
    for i in range(sz):
        if Y[i] == 1:
            error += np.log(pred[i])
            delta.append(1 / pred[i])
    delta = np.array(delta)
    return (-error, -delta)
