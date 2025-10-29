import os
import numpy as np
import matplotlib.pyplot as plt
from src.FFNN import FFNN
from src import utils


def main():
    # Fetch the MNIST dataset
    mnist = utils.load_digit_dataset()

    # Extract data (features) and target (labels)
    X = mnist["data"]
    y = mnist["target"]

    # Format data for our network
    X = X.T / 255.0
    y = np.expand_dims(y, axis=0)
    print(X.shape)
    print(y.shape)

    # nn = FFNN(
    #     network_input_size=
    # )

    # fig, ax = plt.subplots()


if __name__ == "__main__":
    main()
