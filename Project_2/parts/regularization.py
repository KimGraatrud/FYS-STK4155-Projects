import os
import multiprocessing
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from src import utils, costs
from src.FFNN import FFNN


def train(params):
    x, y, der, l = params
    print(f"Starting w/ l = {l:.1e}")

    nn = FFNN(
        network_input_size=x.shape[0],
        layer_output_sizes=[
            50,
            50,
            y.shape[0],
        ],
        activation_funcs=[
            costs.LeakyReLU,
            costs.LeakyReLU,
            costs.one,
            # costs.one,
        ],
        activation_ders=[
            costs.LeakyReLU_der,
            costs.LeakyReLU_der,
            costs.one_der,
            # costs.one,
        ],
        cost_fun=costs.mse,
        cost_der=costs.mse_der,
        eta=5e-2,
        regularization_der=der,
        descent_method="adam",
        decay_rate=(0.99, 0.99),
        lam=l,
    )

    nn.train(x, y, n_iter=1e4)

    print(f"Done w/ l = {l:.1e}, MSE {costs.mse(nn(x)[0], y[0]):.4f}")

    return nn


def main():
    N = 100
    x, y = utils.generate_regression_data(N=N, dim=1, noise_std=0.05)
    x_test, y_test = utils.generate_regression_data(N=100, dim=1, noise_std=0.05)

    # quick scaling
    offset = np.min(y)
    scale = 1 / np.max(y - offset)

    y = (y - offset) * scale
    y_test = (y_test - offset) * scale

    n_lam = 100
    lambdas = np.logspace(-10, -4.7, n_lam)

    with Pool(multiprocessing.cpu_count()) as p:
        L1_params = [(x, y, costs.L1_der, l) for l in lambdas]
        nn_L1 = list(p.map(train, L1_params))

        L2_params = [(x, y, costs.L2_der, l) for l in lambdas]
        nn_L2 = list(p.map(train, L2_params))

    L1_mses = [costs.mse(nn(x_test)[0], y_test[0]) for nn in nn_L1]
    L2_mses = [costs.mse(nn(x_test)[0], y_test[0]) for nn in nn_L2]

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    ax.plot(lambdas, L1_mses, label="L1")
    ax.plot(lambdas, L2_mses, label="L2")

    # Overfitting test MSE, calculated using test.py with no regularization
    ax.axhline(0.004401, lw=0.7, ls="--", c="k", label="no reg.")

    ax.legend()
    ax.set_xscale("log")
    # ax.set_yscale("log")

    ax.set_title("Regularization")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Test MSE")

    fig.savefig(os.path.join(utils.FIGURES_URL, "regularization"))
    plt.close(fig)
