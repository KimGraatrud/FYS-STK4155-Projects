import numpy as np


class FFNN:

    def __init__(
        self,
        network_input_size,
        eta,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
    ):

        self.network_input_size = network_input_size
        self.eta = eta
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.gradient = gradient

    def cost(self, inputs, targets):
        pass

    def _feed_forward_saver(self, inputs, verbose=False):

        # Save inputs internaly
        self.inputs = inputs

        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(layers, activation_funcs):
            layer_inputs.append(a)
            z = W.T @ a + b
            a = activation_func(z)
            zs.append(z)
            if verbose:
                print(f"a: {a.shape}, W: {W.shape}, b: {b.shape}")

        self.layer_inputs = layer_inputs
        self.zs = zs
        self.a = a

    def update_weights(self, layer_grads):
        """
        Updates the weights after a backpropagation cycle.
        """

        for i, (W, b) in enumerate(self.layers):
            self.layers[i][0] = W - self.eta * (layer_grads[i])

    def _create_layers(self):
        layers = []

        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size, 1)
            layers.append((W, b))

            i_size = layer_output_size

        self.layers = layers

    def _Backpropagation(self, inputs):

        # Run to generate layer_input, activation_der, and zs
        _feed_forward_saver(inputs, self.layers, self.activation_funcs)

        layer_grads = [() for layer in self.layers]

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(layers))):
            layer_input = self.layer_inputs[i]
            activation_der = self.activation_ders[i]
            z = self.zs[i]

            if i == len(layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_der(predict, target)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = layers[i + 1]
                dC_da = W @ dC_dz

            dC_dz = dC_da * activation_der(z)
            dC_dW = np.outer(dC_dz, predict)
            dC_db = dC_dz.sum(axis=0)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def train(self, inputs, targets):
        """
        Does backpropagation and updates the weights and biases for all layers
        """
        # Set the train flag, so __call__ cannot be run before.
        self.train = True

        pass

    def __call__(self, test_data):
        """
        Returns the predicted output of a given test input
        """

        if self.trained != True:
            print(
                "NN has not been trained yet, please train the network by calling train(train_data, targets)"
            )
            exit
