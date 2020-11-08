import numpy as np


def mean_squared_error(y, h):
	delta = (y - h)
	error = (np.sum(delta ** 2) * 0.5)
	return error


def binary_crossentropy(y, h):
	if h == 1:
		h -= 1e-9
	if h == 0:
		h += 1e-9
	a = -np.dot(y.T, np.log(h))
	b = -np.dot((1 - y).T, np.log(1 - h))
	return (a + b)

def categorical_crossentropy(y, h):
    if h == 1:
        h -= 1e-9
    if h == 0:
        h += 1e-9
    return (-np.dot(y.T, np.log(h)))