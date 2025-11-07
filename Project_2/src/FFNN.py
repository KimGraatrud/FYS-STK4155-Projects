import numpy as np
from . import utils, costs


class FFNN:
    def __init__(
        self,
        network_input_size,
        eta,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun=costs.mse,
        cost_der=costs.mse_der,
        batch_size=None,
        regularization_der=None,
        descent_method=None,
        delta=1e-7,
        decay_rate=None,
    ):
        """
        descent_method: None, 'rmsprop', or 'adam' (Default None)
        delta: Small constant for numerical stability (Default 1e-7)
        decay_rate: Float if using 'rmsprop' or (Float, Float) if using 'adam' (Default None)
        """
        self.network_input_size = network_input_size
        self.eta = eta
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.batch_size = batch_size
        self.regularization_der = regularization_der
        self.descent_method = descent_method
        self.delta = delta  # for numerical stability in RMSProp and ADAM
        self.decay_rate = decay_rate

        self.trained = False

        self._create_layers()

    def _sample_indices(self, inputs):
        """
        Returns the indices for a sample of inputs
        """
        high = inputs.shape[1]
        if self.batch_size is None:
            return np.arange(high)

        return utils.rng.integers(high, size=self.batch_size)

    def _feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []

        a = inputs

        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = W.T @ a + b[:, None]
            a = activation_func(z)
            zs.append(z)

        layer_inputs.append(a)
        return layer_inputs, zs

    def _compute_update(self, layer_grads, acumltr=None, t=0):
        update = []
        new_acumltr = []
        for i, _ in enumerate(self.layers):
            # Get the gradient by summing over batches
            dW, db = layer_grads[i]
            g = (
                np.sum(dW, axis=2).T / dW.shape[2],
                np.sum(db, axis=1) / db.shape[1],
            )

            # Default
            if self.descent_method is None:
                update.append(
                    (
                        -self.eta * g[0],
                        -self.eta * g[1],
                    )
                )

            # RMSProp
            elif self.descent_method == "rmsprop":
                rho = self.decay_rate
                r = (
                    (np.zeros_like(g[0]), np.zeros_like(g[1]))
                    if acumltr is None
                    else acumltr[i]
                )
                r = (
                    rho * r[0] + (1 - rho) * g[0] ** 2,
                    rho * r[1] + (1 - rho) * g[1] ** 2,
                )
                update.append(
                    (
                        -(self.eta / np.sqrt(self.delta + r[0])) * g[0],
                        -(self.eta / np.sqrt(self.delta + r[1])) * g[1],
                    )
                )
                new_acumltr.append(r)

            # ADAM
            elif self.descent_method == "adam":
                rho1, rho2 = self.decay_rate

                # get s, r
                s = (
                    (np.zeros_like(g[0]), np.zeros_like(g[1]))
                    if acumltr is None
                    else acumltr[i][0]
                )
                r = (
                    (np.zeros_like(g[0]), np.zeros_like(g[1]))
                    if acumltr is None
                    else acumltr[i][1]
                )
                # set s, r
                s = (
                    rho1 * s[0] + (1 - rho1) * g[0],
                    rho1 * s[1] + (1 - rho1) * g[1],
                )
                r = (
                    rho2 * r[0] + (1 - rho2) * g[0] ** 2,
                    rho2 * r[1] + (1 - rho2) * g[1] ** 2,
                )
                # correct bias
                s_hat = (
                    s[0] / (1 - rho1**t),
                    s[1] / (1 - rho1**t),
                )
                r_hat = (
                    r[0] / (1 - rho2**t),
                    r[1] / (1 - rho2**t),
                )
                # compute update
                update.append(
                    (
                        -self.eta * (s_hat[0] / (np.sqrt(r_hat[0]) + self.delta)),
                        -self.eta * (s_hat[1] / (np.sqrt(r_hat[1]) + self.delta)),
                    )
                )
                new_acumltr.append((s, r))
            else:
                raise ValueError(
                    f"'{self.descent_method}' is not a regnized descent method"
                )

        return update, new_acumltr

    def _add_update(self, update):
        """
        Adds update to weights and biases

        update: update with same shape as self.layers
        """
        for i, (W, b) in enumerate(self.layers):
            self.layers[i] = (
                W + update[i][0],
                b + update[i][1],
            )

    def _create_layers(self):
        layers = []

        i_size = self.network_input_size
        for layer_output_size in np.int32(self.layer_output_sizes):
            W = utils.rng.random(size=(i_size, layer_output_size))
            b = utils.rng.random(size=layer_output_size)

            # Scale initial weights so the feed forward doesn't explode
            W = W / np.sum(W)
            b = b / np.sum(b)

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
            (W, b) = self.layers[i]
            z = zs[i]

            if i == len(self.layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_der(predict, targets)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W_prev, _) = self.layers[i + 1]
                dC_da = W_prev @ dC_dz

            dC_dz = dC_da * activation_der(z)
            dC_dW = dC_dz[:, None, :] * layer_input[None, :, :]
            dC_db = dC_dz

            # regularization
            if self.regularization_der is not None:
                dW = self.regularization_der(W).T
                db = self.regularization_der(b)
                dC_dW = dC_dW + dW[:, :, None]
                dC_db = dC_db + db[:, None]

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def train(self, inputs, targets, n_iter=10000, callback=None):
        """
        Does backpropagation and updates the weights and biases for all layers
        """
        # Keep track of accumulated grad. and time step for RMSProp and ADAM
        accumulator = None
        t = 0

        # Train
        for i in range(int(n_iter)):
            t += 1

            # Sample the data
            sample = self._sample_indices(inputs)
            inp = inputs[:, sample]
            tar = targets[:, sample]

            # Backpropagation
            grads = self._Backpropagation(inp, tar)

            # Compute update
            update, accumulator = self._compute_update(grads, accumulator, t)

            # update weights
            self._add_update(update)

            if callback is not None:
                callback(i)

    def __call__(self, test_data):
        """
        Returns the predicted output of a given test input
        """
        a = test_data
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = W.T @ a + b[:, None]
            a = activation_func(z)

        return a
