import numpy as np 


class SGD:

	def __init__(self, lr=0.01, momentum=0.9, decay=1e-8):
		self.lr = lr
		self.momentum = momentum
		self.decay = decay

	def set_init(self, grad_w, grad_b):
		self.upd_w = np.zeros_like(grad_w)
		self.upd_b = np.zeros_like(grad_b)

	def update(self, grad_w, grad_b):
		self.lr -= self.decay
		self.upd_w = (self.momentum * self.upd_w) + ((1 - self.momentum) * grad_w)
		self.upd_b = (self.momentum * self.upd_b) + ((1 - self.momentum) * grad_b)
		return ((self.lr * self.upd_w), (self.lr * self.upd_b))
