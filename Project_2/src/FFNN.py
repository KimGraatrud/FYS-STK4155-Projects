import numpy as np
from . import utils


class FFNN:
    def __init__(
        self,
        network_input_size,
        eta,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun=utils.mse,
        cost_der=utils.mse_der,
        batch_size=None,
    ):

        self.network_input_size = network_input_size
        self.eta = eta
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.batch_size = batch_size

        self.trained = False

        self._create_layers()

    def _sample_indices(self, inputs):
        """
        Returns the indices for a sample of frac% of inputs
        """
        high = inputs.shape[1]
        if self.batch_size is None:
            return np.arange(high)

        return utils.rng.integers(high, size=self.batch_size)

    def _feed_forward_saver(self, inputs, verbose=False):
        layer_inputs = []
        zs = []

        a = inputs

        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = W.T @ a + b[:, None]
            a = activation_func(z)
            zs.append(z)
            if verbose:
                print(f"a: {a.shape}, W: {W.shape}, b: {b.shape}")

        layer_inputs.append(a)
        return layer_inputs, zs

    def _update_weights(self, layer_grads):
        """
        Updates the weights after a backpropagation cycle.
        """

        for i, (W, b) in enumerate(self.layers):
            self.layers[i] = (
                W - self.eta * np.sum((layer_grads[i][0]), axis=2).T,
                b - self.eta * np.sum((layer_grads[i][1]), axis=1),
            )

    def _create_layers(self):
        layers = []

        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = utils.rng.random(size=(i_size, layer_output_size))
            b = utils.rng.random(size=layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size

        self.layers = layers

    def _Backpropagation(self, inputs, targets):
        # Run to generate layer_input, activation_der, and zs
        layer_inputs, zs = self._feed_forward_saver(inputs)

        layer_grads = [() for layer in self.layers]

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input = layer_inputs[i]
            predict = layer_inputs[i + 1]
            activation_der = self.activation_ders[i]
            z = zs[i]

            if i == len(self.layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_der(predict, targets)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = self.layers[i + 1]
                dC_da = W @ dC_dz

            dC_dz = dC_da * activation_der(z)
            dC_dW = dC_dz[:, None, :] * layer_input[None, :, :]
            dC_db = dC_dz

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def train(self, inputs, targets, n_iter=10000):
        """
        Does backpropagation and updates the weights and biases for all layers
        """

        # Set the train flag, so __call__ doesn't warn.
        self.trained = True

        for i in range(int(n_iter)):
            # Sample the data
            sample = self._sample_indices(inputs)
            inp = inputs[:, sample]
            tar = targets[:, sample]

            # Backpropagation
            grads = self._Backpropagation(inp, tar)

            # updat weights
            self._update_weights(grads)

    def __call__(self, test_data):
        """
        Returns the predicted output of a given test input
        """
        if not self.trained:
            print(
                "NN has not been trained yet, please train the network by calling train(train_data, targets)"
            )

        a = test_data
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = W.T @ a + b[:, None]
            a = activation_func(z)

        return a
