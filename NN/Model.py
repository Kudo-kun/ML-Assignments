import Losses
import numpy as np
from math import sqrt
import NonLinearizers
import matplotlib.pyplot as plt
from tqdm import tqdm


class nn_sequential_model:

    def __init__(self):
        self.W = []
        self.B = []
        self.layers = []
        self.nLayers = 0

    def add_layer(self, layer, initialization="uniform", seed=5):
        """
        add a new layer to the model
        i.e. append a new set of weights
        and biases
        """
        np.random.seed(seed)
        self.nLayers += 1
        self.layers.append(layer)
        if self.nLayers > 1:
            n = self.layers[-1].units
            m = self.layers[-2].units
            if initialization == "normal":
                self.W.append(np.random.randn(m, n))
                self.B.append(np.random.randn(n))
            elif initialization == "uniform":
                self.W.append(np.random.random((m, n)))
                self.B.append(np.random.random(n))

    def _feed_forward(self, X):
        """
        perform a forward pass and
        return the final predictions
        """
        act, pre_act = [], []
        z = (self.layers[0].activation(X))
        pre_act.append(X)
        act.append(z)
        for i in range(1, self.nLayers):
            a = np.dot(self.W[i - 1].T, z)
            a += self.B[i - 1]
            z = self.layers[i].activation(a)
            pre_act.append(a)
            act.append(z)
        return np.array(z), np.array(pre_act), np.array(act)

    def _back_prop(self, pred, Y, pre_act, act):
        """
        back propagates the error
        and tweaks the weights and 
        biases in the network
        """
        deltas, grad_w, grad_b = [], [], []
        if self.loss == "mse":
            error = Losses.mean_squared_error(Y, pred)
        elif self.loss == "binary_crossentropy":
            error = Losses.binary_crossentropy(Y, pred)

        delta = (pred - Y)
        for i in range(self.nLayers - 1, -1, -1):
            if i < (self.nLayers - 1):
                delta = ed * self.layers[i].activation(pre_act[i], derv=True)
                grad = np.outer(delta, act[i + 1])
                grad_w.append(grad)
            if i > 0:
                grad_b.append(delta)
                ed = np.dot(self.W[i - 1], delta)

        grad_w = np.array(grad_w[::-1])
        grad_b = np.array(grad_b[::-1])
        self.W -= self.lr * grad_w
        self.B -= self.lr * grad_b
        return error

    def compile(self, loss, epochs=100, lr=0.01):
        self.lr = lr
        self.loss = loss
        self.epochs = epochs + 1

    def fit(self, X_train, Y_train, plot_freq=None):
        """
        performs an SGD on the data.
        A single data point is chosen
        and the a complete cycle, i.e.
        forward pass and a backprop are
        completed.
        """
        ep, err = [], []
        for _ in tqdm(range(0, self.epochs), ncols=100):
            i = np.random.randint(0, len(X_train))
            pred, pre_act, act = self._feed_forward(X_train[i])
            error = self._back_prop(pred=pred,
                                    Y=Y_train[i],
                                    pre_act=pre_act,
                                    act=act)

            if plot_freq is not None and (_ % plot_freq) == 0:
                error = round(error, 2)
                err.append(error)
                ep.append(_)
                # print("epoch: {}\tloss: {}".format(_, error))

        if plot_freq != None:
            plt.xlabel("epochs")
            plt.ylabel("cost")
            plt.plot(ep, err)
            plt.show()
        return

    def get_params(self):
        """
        return the parameters
        of the neural network
        """
        return (self.W, self.B)

    def predict(self, X_test):
        """
        returns the predictions
        for the given testing points
        based on the trained weights
        and biases. It's expected the
        model is trained beforehand
        """
        result = []
        for x in X_test:
            err, _, _ = self._feed_forward(x)
            result.append(err)
        return np.array(result)

    def evaluate(self, pred, Y_test, verbose=False):
        """
        prints the necessary metrics
        for the corresponding prediction
        """
        if self.loss == "mse":
            error, _ = Losses.mean_squared_error(Y_test, pred)
            mse = (error / pred.shape[0])
            print("final mse: {}".format(mse))
        elif self.loss == "binary_crossentropy":
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

            return accuracy
