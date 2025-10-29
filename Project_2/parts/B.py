import os
import numpy as np
import matplotlib.pyplot as plt
from src import utils
from src.FFNN import FFNN


def one(x):
    return x


def one_der(x):
    return np.ones_like(x)


def main():
    dim = 2
    N = 10000
    x, y = utils.generate_regression_data(N=N, dim=dim, noise_std=0.05)

    # quick scaling
    y -= np.min(y)
    y /= np.max(y)

    nn = FFNN(
        network_input_size=x.shape[0],
        layer_output_sizes=[5, 5, y.shape[0]],
        activation_funcs=[utils.sigmoid, utils.ReLU, utils.sigmoid],
        activation_ders=[utils.sigmoid_der, utils.ReLU_der, utils.sigmoid_der],
        cost_fun=utils.mse,
        cost_der=utils.mse_der,
        eta=1e-2,
        batch_size=200,
    )

    nn.train(x, y, n_iter=3e5)

    print("trained!")

    # instead of splitting into train & test, right now it's
    # easier to just create new data for testing
    x_test, y_test = utils.generate_regression_data(N=30, dim=dim, noise_std=0.05)

    # quick scaling
    y_test -= np.min(y_test)
    y_test /= np.max(y_test)

    y_pred = nn(x_test)

    if dim == 2:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(x_test[0], x_test[1], y_test[0], s=1, alpha=1)
        ax.scatter(x_test[0], x_test[1], y_pred[0], s=1, alpha=1)
        plt.show()
    else:
        fig, ax = plt.subplots()
        ax.scatter(x_test[0], y_test[0], s=0.5)
        ax.scatter(x_test[0], y_pred[0])
        fig.savefig(os.path.join(utils.FIGURES_URL, "test"))


if __name__ == "__main__":
    main()
