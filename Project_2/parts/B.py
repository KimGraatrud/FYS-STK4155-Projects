import os
import numpy as np
import matplotlib.pyplot as plt
from src import utils, costs
from src.FFNN import FFNN


def one(x):
    return x


def one_der(x):
    return np.ones_like(x)


def main():
    dim = 1
    N = 10000
    x, y = utils.generate_regression_data(N=N, dim=dim, noise_std=0.05)

    # quick scaling
    y -= np.min(y)
    y /= np.max(y)

    # der = None
    # der = lambda X: costs.L1_der(X, lam=3e-5)
    der = lambda X: costs.L2_der(X, lam=1e-4)

    nn = FFNN(
        network_input_size=x.shape[0],
        layer_output_sizes=[
            5,
            y.shape[0],
        ],
        activation_funcs=[
            costs.sigmoid,
            # costs.ReLU,
            costs.sigmoid,
        ],
        activation_ders=[
            costs.sigmoid_der,
            # costs.ReLU_der,
            costs.sigmoid_der,
        ],
        cost_fun=costs.mse,
        cost_der=costs.mse_der,
        eta=2e-1,
        batch_size=100,
        regularization_der=der,
    )

    def callback(i):
        if i % 1000 == 0:
            print(i)

    nn.train(x, y, n_iter=1e3, callback=callback)

    # instead of splitting into train & test, right now it's
    # easier to just create new data for testing
    x_test, y_test = utils.generate_regression_data(N=100, dim=dim, noise_std=0.05)

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
