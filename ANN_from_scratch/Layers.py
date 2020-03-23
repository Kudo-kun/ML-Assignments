import NonLinearizers


class dense_layer:

    def __init__(self, non, activation):
        """
        arguments:
        non -- number of nodes in that layer
        activation -- string describing the activation associated with this layer
        """
        self.non = non
        if activation == "sigmoid":
            self.activation = NonLinearizers.sigmoid
        elif activation == "tanh":
            self.activation = NonLinearizers.tanh
        elif activation == "linear":
            self.activation = NonLinearizers.linear
        elif activation == "softmax":
            self.activation = NonLinearizers.softmax
        elif activation == "relu":
            self.activation = NonLinearizers.relu
