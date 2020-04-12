import numpy as np 


class SGD:

	def __init__(self, lr=0.01, momentum=0.9, decay=1e-8):
		self.lr = lr
		self.momentum = momentum
		self.decay = decay

	def set_init(self, grad_w, grad_b):
		self.hist_w = np.zeros_like(grad_w)
		self.hist_b = np.zeros_like(grad_b)

	def update(self, grad_w, grad_b):
		self.lr -= self.decay
		self.hist_w = (self.momentum * self.hist_w) + ((1 - self.momentum) * grad_w)
		self.hist_b = (self.momentum * self.hist_b) + ((1 - self.momentum) * grad_b)
		dw = self.lr * self.hist_w
		db = self.lr * self.hist_b
		return (dw, db)


class RMSprop:

	def __init__(self, lr=0.01, beta=0.9):
		self.lr = lr
		self.beta = beta	

	def set_init(self, grad_w, grad_b):
		self.hist_w = np.zeros_like(grad_w)
		self.hist_b = np.zeros_like(grad_b)

	def update(self, grad_w, grad_b):
		self.hist_w = (self.beta * self.hist_w) + ((1 - self.beta) * (grad_w ** 2))
		self.hist_b = (self.beta * self.hist_b) + ((1 - self.beta) * (grad_b ** 2))
		hist_w = self.hist_w
		hist_b = self.hist_b
		for i in range(hist_w.shape[0]):
			hist_w[i] = np.sqrt(hist_w[i]) + 1e-9
		for i in range(hist_w.shape[0]):
			hist_b[i] = np.sqrt(hist_b[i]) + 1e-9
		dw = self.lr * (grad_w / hist_w)
		db = self.lr * (grad_b / hist_b)
		return (dw, db)

class Adam:

	def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2

	def set_init(self, grad_w, grad_b):
		self.t = 0
		self.hist_mw = np.zeros_like(grad_w)
		self.hist_vw = np.zeros_like(grad_w)
		self.hist_mb = np.zeros_like(grad_b)
		self.hist_vb = np.zeros_like(grad_b)

	def update(self, grad_w, grad_b):
		self.t += 1
		self.hist_mw = (self.beta1 * self.hist_mw) + ((1 - self.beta1) * grad_w)
		self.hist_mb = (self.beta1 * self.hist_mb) + ((1 - self.beta1) * grad_b)
		self.hist_vw = (self.beta2 * self.hist_vw) + ((1 - self.beta2) * (grad_w ** 2))
		self.hist_vb = (self.beta2 * self.hist_vb) + ((1 - self.beta2) * (grad_b ** 2))
		m_hat_w = self.hist_mw / (1 - (self.beta1 ** self.t))
		m_hat_b = self.hist_mb / (1 - (self.beta1 ** self.t))
		v_hat_w = self.hist_vw / (1 - (self.beta2 ** self.t))
		v_hat_b = self.hist_vb / (1 - (self.beta2 ** self.t))
		for i in range(v_hat_w.shape[0]):
			v_hat_w[i] = np.sqrt(v_hat_w[i]) + 1e-9
		for i in range(v_hat_b.shape[0]):
			v_hat_b[i] = np.sqrt(v_hat_b[i]) + 1e-9
		dw = self.lr * (m_hat_w / v_hat_w)
		db = self.lr * (m_hat_b / v_hat_b)
		return (dw, db)
