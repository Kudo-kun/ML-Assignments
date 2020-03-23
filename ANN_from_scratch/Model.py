import numpy as np
import Losses
import NonLinearizers
import matplotlib.pyplot as plt


class nn_sequential_model:

    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []
        self.no_of_layers = 0

    def add_layer(self, layer):
        """
        add a new layer to the model
        i.e. append a new set of weights
        and biases
        """
        self.layers.append(layer)
        self.no_of_layers += 1
        if self.no_of_layers > 1:
            n = self.layers[-1].non
            m = self.layers[-2].non
            self.weights.append(np.random.randn(m, n))
            self.biases.append(np.random.randn(n))

    def feed_forward(self, X):
        """
        perform a forward pass and
        return the final predictions
        """
        act = []
        z = self.layers[0].activation(X)
        act.append(z)
        for i in range(1, self.no_of_layers):
            a = np.dot(self.weights[i - 1].T, z)
            a += np.array(self.biases[i - 1])
            z = self.layers[i].activation(a)
            act.append(z)
        return (np.array(z), np.array(act))

    def back_prop(self, lr, pred, Y, act, loss):
        """
        back propagates the error
        and tweaks the weights
        """
        deltas = []
        if loss == "MSE":
            err = Losses.MSE(Y, pred, derivative=True)

        curr_error = (np.sum(err ** 2) * 0.5)
        for i in range(self.no_of_layers - 1, -1, -1):
            delta = err * self.layers[i].activation(act[i], derv=True)
            if i < (self.no_of_layers - 1):
                # print("delta: ", delta)
                # print("act: ", act[i + 1])
                grad = np.outer(delta.T, act[i + 1])
                self.weights[i] -= lr * grad
            if i > 0:
                self.biases[i - 1] -= lr * delta
                err = np.dot(self.weights[i - 1], delta)
        return curr_error

    def train(self, X_train, Y_train, epochs, lr=None, plot_freq=None):
        """
        performs an SGD on the data.
        A single data point is chosen
        and the a complete cycle, i.e.
        forward pass and a backprop are
        completed.
        """
        if lr == None:
            lr = 0.01
        ep, err = [], []
        self.weights = np.array(self.weights)
        for _ in range(epochs):
            print("epoch: " + str(_), end='\t')
            idx = np.random.randint(0, len(X_train))
            pred, act = self.feed_forward(X_train[idx])
            error = self.back_prop(lr=lr,
                                   pred=pred,
                                   Y=Y_train[idx],
                                   act=act,
                                   loss="MSE")
            print(error)
            if plot_freq != None and (_ % plot_freq) == 0:
                ep.append(_)
                err.append(error)
        print("training complete!")
        if plot_freq != None:
            plt.xlabel("epochs")
            plt.ylabel("error")
            plt.plot(ep, err)
            plt.show()
        return

    def get_parameters():
        return (self.weights, self.biases)

    def predict(self, X_test):
        result = []
        for x in X_test:
            err, _ = self.feed_forward(x)
            result.append(err)
        return np.array(result)

    def evaluate(self, pred, Y_test):
        print("y_pred: ", pred)
        print("y_test: ", np.array(Y_test))
