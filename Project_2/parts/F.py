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

    # Split into training and testing
    X_train, X_test, y_train, y_test = utils.train_test_split(X, y)

    # Format data for our network
    X_train = X_train.T / 255.0
    X_test = X_test.T / 255.0
    y_train = utils.onehot(np.int32(y_train))
    y_test = utils.onehot(np.int32(y_test))

    nn = ClassifierNN(
        network_input_size=X_train.shape[0],
        classes=y_train.shape[0],
        layer_output_sizes=[
            25,
            15,
        ],
        activation_funcs=[utils.ReLU, utils.ReLU],
        activation_ders=[utils.ReLU_der, utils.ReLU_der],
        eta=1e-3,
        batch_size=150,
    )

    # quick function for reporting
    def train_callback(i):
        if i % 500 == 0:
            train = utils.accuracy(nn(X_train), y_train)
            test = utils.accuracy(nn(X_test), y_test)
            print(f"train: {train:.4f} test: {test:.4f} iter: {i}")

    nn.train(X_train, y_train, n_iter=1e6, callback=train_callback)


if __name__ == "__main__":
    main()
