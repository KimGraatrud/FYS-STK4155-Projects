import numpy as np
from src.FFNN import FFNN
from src import utils


class ClassifierNN(FFNN):
    def __init__(self, *args, classes, **kwargs):
        # Initialize the usual variables
        FFNN.__init__(self, *args, **kwargs)
        self.classes = classes

        # Cost is handled internaly in classification
        self.cost_der = None
        self.cost_fun = None

        # Add a final layer of softmax and recreate layers
        self.activation_funcs.append(utils.softmax)
        self.layer_output_sizes.append(classes)
        self._create_layers()

    def _Backpropagation(self, inputs, targets):
        # Run to generate layer_input, activation_der, and zs
        layer_inputs, zs = self._feed_forward_saver(inputs)

        layer_grads = [() for layer in self.layers]

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input = layer_inputs[i]
            predict = layer_inputs[i + 1]
            z = zs[i]

            if i == len(self.layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_dz = predict - targets

            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                activation_der = self.activation_ders[i]
                (W, b) = self.layers[i + 1]
                dC_da = W @ dC_dz
                dC_dz = dC_da * activation_der(z)
            dC_dW = dC_dz[:, None, :] * layer_input[None, :, :]
            dC_db = dC_dz

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads
