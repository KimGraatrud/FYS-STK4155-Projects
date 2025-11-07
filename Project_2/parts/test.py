import os
import numpy as np
import matplotlib.pyplot as plt
from src import utils, costs
from src.FFNN import FFNN


def main():
    dim = 2
    N = 100
    x, y = utils.generate_regression_data(N=N, dim=dim, noise_std=0.05)

    # # quick scaling
    y -= np.min(y)
    y /= np.max(y)

    der = None
    # der = lambda X: costs.L1_der(X, lam=1e-5)
    # der = lambda X: costs.L2_der(X, lam=1e-4)

    nn = FFNN(
        network_input_size=x.shape[0],
        layer_output_sizes=[
            # 10,
            16,
            16,
            # 10,
            y.shape[0],
        ],
        activation_funcs=[
            # costs.ReLU,
            costs.LeakyReLU,
            costs.sigmoid,
            # costs.ReLU,
            # costs.LeakyReLU,
            # costs.one,
            costs.one,
            # costs.ReLU,
        ],
        activation_ders=[
            costs.LeakyReLU_der,
            costs.sigmoid_der,
            # costs.ReLU_der,
            # costs.ReLU_der,
            # costs.LeakyReLU_der,
            # costs.one_der,
            costs.one_der,
            # costs.ReLU_der,
        ],
        cost_fun=costs.mse,
        cost_der=costs.mse_der,
        eta=1e-2,
        batch_size=64,
        regularization_der=der,
        descent_method="adam",
        # decay_rate=0.9,
        decay_rate=(0.9, 0.999),
    )

    # instead of splitting into train & test, right now it's
    # easier to just create new data for testing
    x_test, y_test = utils.generate_regression_data(N=30, dim=dim, noise_std=0.05)

    # # quick scaling
    y_test -= np.min(y_test)
    y_test /= np.max(y_test)

    def callback(i):
        if i % 1000 == 0:
            print(
                i,
                f"MSE train {costs.mse(nn(x)[0], y[0]):.4f}",
                f"test {costs.mse(nn(x_test)[0], y_test[0]):.4f}",
            )

    try:
        nn.train(x, y, n_iter=2e5, callback=callback)
    except KeyboardInterrupt:
        print("\nplotting anyway")

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
        ax.legend()
        fig.savefig(os.path.join(utils.FIGURES_URL, "test"))
