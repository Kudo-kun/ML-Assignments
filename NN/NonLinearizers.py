import numpy as np


def softmax(s, derv=False):
    if not derv:
        p = np.exp(s)
        return (p/np.sum(p))
    else:
        pass

def sigmoid(s, derv=False):
    x = (1 / (1 + np.exp(-s)))
    if not derv:
        return x
    else:
        return (x * (1 - x))

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
        return np.maximum(0, s)
    else:
        f = lambda x: (1 if x >= 0 else 0)
        return np.array([f(i) for i in s])

def leaky_relu(s, derv=False):
    a = 0.3
    if not derv:
        f = lambda x: (x if x >= 0 else a*x)
        return np.array([f(i) for i in s])
    else:
        f = lambda x: (1 if x >= 0 else a)
        return np.array([f(i) for i in s])
