import NonLinearizers


class dense_layer:

    def __init__(self, units, activation="linear"):
        """
        arguments:
        units -- number of nodes in that layer
        activation -- string describing the 
                      activation associated 
                      with this layer
        """
        self.units = units
        if activation == "sigmoid":
            self.activation = NonLinearizers.sigmoid
        elif activation == "tanh":
            self.activation = NonLinearizers.tanh
        elif activation == "linear":
            self.activation = NonLinearizers.linear
        elif activation == "relu":
            self.activation = NonLinearizers.relu
        elif activation == "leaky_relu":
            self.activation = NonLinearizers.leaky_relu
