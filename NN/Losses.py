import numpy as np


def mean_squared_error(y, h):
	delta = (y - h)
	error = (np.sum(delta ** 2) * 0.5)
	return error


def binary_crossentropy(y, h):
	a = -np.dot(y.T, np.log(h + 1e-9))
	b = -np.dot((1 - y).T, np.log(1 - h + 1e-9))
	return (a + b)
