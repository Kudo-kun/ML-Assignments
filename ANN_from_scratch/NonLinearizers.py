import numpy as np


def sigmoid(s, derv=False):
    x = 1 / (1 + np.exp(-s))
    if not derv:
        return x
    else:
        return x * (1 - x)


def tanh(s, derv=False):
    x = ((np.exp(2 * s) - 1) / (np.exp(2 * s) + 1))
    if not derv:
        return x
    else:
        return (1 - (x ** 2))


def linear(s, derv=False):
    if not derv:
        return s
    else:
        return 1


def relu(s, derv=False):
    if not derv:
        return np.array([max(0, i) for i in s])
    else:
        f = lambda x: (1 if x > 0 else 0)
        return np.array([f(i) for i in s])


def softmax(s, derv=False):
    if not derv:
        exps = np.exp(s)
        return (exps / np.sum(exps))
