import numpy as np
from math import sqrt
import NonLinearizers
from random import randrange
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import Losses


class nn_sequential_model:

    def __init__(self):
        self.W = []
        self.B = []
        self.layers = []
        self.nLayers = 0

    def add_layer(self, layer, initialization="uniform", seed=0):
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
        self.layers[0].preactv = X
        self.layers[0].actv = self.layers[0].activation(X)
        for i in range(1, self.nLayers):
            z = self.layers[i - 1].actv
            a = np.dot(self.W[i - 1].T, z)
            a += self.B[i - 1]
            z = self.layers[i].activation(a)
            self.layers[i].preactv = a
            self.layers[i].actv = z
        return self.layers[-1].actv

    def _back_prop(self, pred, Y):
        """
        back propagates the error
        and tweaks the weights and 
        biases in the network
        """
        if self.loss == "mse":
            error = Losses.mean_squared_error(Y, pred)
        elif self.loss == "binary_crossentropy":
            error = Losses.binary_crossentropy(Y, pred)
        elif self.loss == "categorical_crossentropy":
            error = Losses.categorical_crossentropy(Y, pred)

        db = (pred - Y)
        for i in range(self.nLayers - 1, -1, -1):
            if i < self.nLayers - 1:
                a = self.layers[i].preactv
                z = self.layers[i + 1].actv
                db = dz * self.layers[i].activation(a, derv=True)
                dw = np.outer(db, z)
                self.W[i] -= self.lr * dw
            if i > 0:
                self.B[i - 1] -= self.lr * db
                dz = np.dot(self.W[i - 1], db)
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
        it, err = [], []
        for _ in range(self.epochs):
            i = randrange(len(X_train) - 1)
            pred = self._feed_forward(X_train[i])
            error = self._back_prop(pred=pred, Y=Y_train[i])
            if plot_freq is not None and not (_ % plot_freq):
                error = round(error, 2)
                err.append(error)
                it.append(_)
                # print("epoch: {}\tloss: {}".format(_, error))

        if plot_freq != None:
            plt.xlabel("epochs")
            plt.ylabel("cost")
            plt.plot(it, err, color='r')
            plt.show()
        return error

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
            result.append(self._feed_forward(x))
        return np.array(result)

    def evaluate(self, pred, Y_test):
        """
        prints the necessary metrics
        for the corresponding prediction
        """
        if self.loss == "mse":
            error, _ = Losses.mean_squared_error(Y_test, pred)
            mse = (error / pred.shape[0])
            print("final mse: {}".format(mse))
        elif self.loss == "binary_crossentropy":
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            accuracy = (np.sum(pred == Y_test))/len(Y_test)
            print(round(accuracy, 3))
        elif self.loss == "categorical_crossentropy":
            pass


