import os
import numpy as np
import matplotlib.pyplot as plt
from src.FFNN import FFNN
from src.ClassifierNN import ClassifierNN
from src import utils


def main():
    # Fetch the MNIST dataset
    mnist = utils.load_openml_dataset()

    # Extract data (features) and target (labels)
    X = mnist["data"]
    y = mnist["target"]

    # Format data for our network
    X = X.T / 255.0
    y = utils.onehot(np.int32(y))

    # TODO: train/test split

    nn = ClassifierNN(
        network_input_size=X.shape[0],
        classes=y.shape[0],
        layer_output_sizes=[30, 30, 15],
        activation_funcs=[utils.sigmoid, utils.ReLU, utils.sigmoid],
        activation_ders=[utils.sigmoid_der, utils.ReLU_der, utils.sigmoid_der],
        eta=1e-4,
        batch_size=100,
    )

    nn.train(X, y, n_iter=1e3)

    # fig, ax = plt.subplots()


if __name__ == "__main__":
    main()
