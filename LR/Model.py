
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class LogisticRegression:

    def __init__(self, nfeatures, seed=5, initialization="normal"):
        np.random.seed(seed)
        self.features = nfeatures
        if initialization == "normal":
            self.weights = np.random.randn(nfeatures + 1)
        elif initilization == "uniform":
            self.weights = np.random.random(nfeatures + 1)
        self.weights.resize(self.weights.shape[0], 1)

    def _sigmoid(self, s):
        x = 1 / (1 + np.exp(-s))
        return x

    def predict(self, X, W):
        h = self._sigmoid(np.dot(X, W))
        h.resize(h.shape[0], 1)
        return h

    def _loss(self, y, h):
        a = -np.dot(y.T, np.log(h))
        b = -np.dot((1 - y).T, np.log(1 - h))
        return (a + b)[0]

    def get_params(self):
        return self.weights

    def fit(self, X, Y, xval=None, yval=None, plot_freq=None):
        if self.penalty == None:
            ep, err = [], []
            for _ in tqdm(range(self.epochs), ncols=100):
                h = self.predict(X, self.weights)
                log_loss = self._loss(Y, h) / h.shape[0]
                gradients = np.dot(X.T, (h - Y))
                self.weights -= (self.lr * gradients)
                if plot_freq != None and (_ % plot_freq) == 0:
                    ep.append(_)
                    err.append(log_loss)

            if plot_freq != None:
                plt.xlabel("epochs")
                plt.ylabel("log_loss")
                plt.plot(ep, err)
                plt.show()

        elif self.penalty == "L2":
            max_met, opt_beta = 0, 0
            for l2 in np.linspace(0.0, 1.5, 15):
                print("checking for lambda: {}".format(l2))
                Lmet, W = self._fit_with_L2(X, Y,
                                            xval=xval,
                                            yval=yval,
                                            beta=l2)
                print(Lmet)
                if Lmet >= max_met:
                    max_met = Lmet
                    self.weights = W
                    opt_beta = l2
            print("optimal L2 lambda: {}".format(opt_beta))

        elif self.penalty == "L1":
            max_met, opt_beta = 0, 0
            for l1 in np.linspace(0.0, 1.5, 15):
                print("checking for lamda: {}".format(l1))
                Lmet, W = self._fit_with_L1(X, Y,
                                            xval=xval,
                                            yval=yval,
                                            beta=l1)
                print(Lmet)
                if Lmet >= max_met:
                    max_met = Lmet
                    self.weights = W
                    opt_beta = l1
            print("optimal L1 lambda: {}".format(opt_beta))

    def _fit_with_L2(self, X, Y, xval, yval, beta):
        W = np.random.random(self.features + 1)
        W.resize(W.shape[0], 1)
        for _ in range(self.epochs):
            h = self.predict(X, W)
            log_loss = self._loss(Y, h) + (0.5 * beta *
                                           np.dot(W.T, W))
            gradients = np.dot(X.T, (h - Y)) + (beta * W)
            W -= (self.lr * gradients)

        h = self.predict(xval, W)
        Lmet = self.evaluate(h, yval)
        return (Lmet, W)

    def _fit_with_L1(self, X, Y, xval, yval, beta):
        W = np.random.random(self.features + 1)
        W.resize(W.shape[0], 1)
        for _ in range(self.epochs):
            h = self.predict(X, W)
            log_loss = self._loss(Y, h) + (0.5 * beta *
                                           np.absolute(W))

            gradients = np.dot(X.T, (h - Y)) + (beta * np.sign(W))
            W -= (self.lr * gradients)

        h = self.predict(xval, W)
        Lmet = self.evaluate(h, yval)
        return (Lmet, W)

    def compile(self, epochs, penalty=None, learning_rate=0.01, metrics=None):
        self.penalty = penalty
        self.lr = learning_rate
        self.epochs = epochs + 1
        self.metrics = metrics

    def evaluate(self, pred, Y_test, verbose=False):
        cm = np.array([[0, 0], [0, 0]])
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        zipped = np.array(list(zip(pred, Y_test)))
        for (x, y) in zipped:
            cm[int(x)][int(y)] += 1

        accuracy = ((cm[0][0] + cm[1][1]) / np.sum(cm)) * 100
        recall = (cm[1][1] / (cm[1][1] + cm[0][1])) * 100
        precision = (cm[1][1] / np.sum(cm[1])) * 100
        fscore = 2 * ((precision * recall) / (precision + recall))
        if verbose:
            print("tp = {}, tn = {}, fp = {}, fn = {}".format(
                cm[1][1], cm[0][0], cm[1][0], cm[0][1]))
            print("final accuracy: {}".format(accuracy))
            print("final recall: {}".format(recall))
            print("final precision: {}".format(precision))
            print("final fscore: {}".format(fscore))

        if self.metrics == "accuracy":
            return accuracy
        if self.metrics == "fscore":
            return fscore
        if self.metrics == "recall":
            return recall
        if self.metrics == "precision":
            return precision
