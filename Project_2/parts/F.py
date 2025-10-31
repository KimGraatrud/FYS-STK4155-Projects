import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.FFNN import FFNN
from src.ClassifierNN import ClassifierNN
from src import utils, costs


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

    # der = None
    # der = lambda X: costs.L1_der(X, lam=3e-5)
    der = lambda X: costs.L2_der(X, lam=1e-4)

    nn = ClassifierNN(
        network_input_size=X_train.shape[0],
        classes=y_train.shape[0],
        layer_output_sizes=[
            10,
            # 15,
        ],
        activation_funcs=[
            costs.ReLU,
            # costs.ReLU,
        ],
        activation_ders=[
            costs.ReLU_der,
            # costs.ReLU_der,
        ],
        eta=1e-3,
        batch_size=150,
        regularization_der=der,
    )

    # quick function for reporting
    def train_callback(i):
        if i % 500 == 0:
            print(i)

    #         train = utils.accuracy(nn(X_train), y_train)
    #         test = utils.accuracy(nn(X_test), y_test)
    #         print(f"train: {train:.4f} test: {test:.4f} iter: {i}")

    nn.train(
        X_train,
        y_train,
        n_iter=5e3,
        callback=train_callback,
    )


if __name__ == "__main__":
    main()
