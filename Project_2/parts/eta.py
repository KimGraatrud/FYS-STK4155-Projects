import os
import multiprocessing
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from src import utils, costs
from src.ClassifierNN import ClassifierNN
from src.FFNN import FFNN


def _regress(params):
    eta, n_iter, data, test = params
    X, y = data
    X_test, y_test = test

    # quick scaling
    offset = np.min(y)
    scale = 1 / np.max(y - offset)

    y = (y - offset) * scale
    y_test = (y_test - offset) * scale

    nn = FFNN(
        network_input_size=X.shape[0],
        layer_output_sizes=[16, y.shape[0]],
        activation_funcs=[costs.LeakyReLU, costs.one],
        activation_ders=[costs.LeakyReLU_der, costs.one_der],
        eta=eta,
        batch_size=32,
        regularization_der=None,
        descent_method="adam",
        decay_rate=(0.9, 0.999),
    )

    nn.train(X, y, n_iter=n_iter)

    print(f"eta: {eta:.1e} regress")

    pred = nn(X_test)

    return costs.mse(pred[0], y_test[0])


def _classify(params):
    eta, n_iter, data = params
    X, X_test, y, y_test = utils.prep_classification_data(data)

    nn = ClassifierNN(
        network_input_size=X.shape[0],
        classes=y.shape[0],
        layer_output_sizes=[],
        activation_funcs=[],
        activation_ders=[],
        eta=eta,
        batch_size=32,
        regularization_der=None,
        descent_method="adam",
        decay_rate=(0.9, 0.999),
    )

    nn.train(X, y, n_iter=n_iter)

    pred = nn(X_test)

    print(f"eta: {eta:.1e} classy")

    return utils.accuracy(pred, y_test)


def main():
    n_etas = 100
    etas = np.logspace(-5, 0, n_etas)
    n_iters = [3e4, 1e2]

    with Pool(multiprocessing.cpu_count()) as p:
        oned_scores = []
        twod_scores = []
        mnist_scores = []
        fashion_scores = []
        for n in n_iters:
            # 1D Runge
            data = utils.generate_regression_data(noise_std=0.05, dim=1)
            test = utils.generate_regression_data(N=100, noise_std=0.05, dim=1)
            oned_params = [(eta, n, data, test) for eta in etas]
            oned_scores.append(list(p.map(_regress, oned_params)))

            # 2D Runge
            data = utils.generate_regression_data(N=100, noise_std=0.05, dim=2)
            test = utils.generate_regression_data(N=100, noise_std=0.05, dim=2)
            twod_params = [(eta, n, data, test) for eta in etas]
            twod_scores.append(list(p.map(_regress, twod_params)))

            # MNIST
            mnist = utils.load_openml_dataset()
            mnist_params = [(eta, n, mnist) for eta in etas]
            mnist_scores.append(list(p.map(_classify, mnist_params)))

            # Fashion
            fashion = utils.load_openml_dataset(dataset="fashion-mnist")
            fashion_params = [(eta, n, fashion) for eta in etas]
            fashion_scores.append(list(p.map(_classify, fashion_params)))

    # Plot everything
    # colors = ["lightcoral", "crimson", "darkred"]
    fig, axs = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(utils.APS_COL_W, 1.3 * utils.APS_COL_W),
    )

    ax = axs[0]
    ax.plot(etas, oned_scores[0], label="1D Runge", c="rosybrown")
    ax.plot(etas, oned_scores[1], c="rosybrown", ls="--")
    ax.plot(etas, twod_scores[0], label="2D Runge", c="k")
    ax.plot(etas, twod_scores[1], c="k", ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("Test MSE")
    ax.set_title("Regression")
    ax.legend()

    ax = axs[1]
    ax.plot(etas, mnist_scores[0], label=f"MNIST", c="purple")
    ax.plot(etas, mnist_scores[1], ls="--", c="purple")
    ax.plot(etas, fashion_scores[0], label=f"Fashion", c="goldenrod")
    ax.plot(etas, fashion_scores[1], ls="--", c="goldenrod")

    ax.set_title("Classification")
    ax.set_xlabel(r"Learning Rate $\eta$")
    ax.set_xscale("log")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.suptitle("Learning Rate Response by Dataset")

    fig.savefig(os.path.join(utils.FIGURES_URL, "eta"))
    plt.close(fig)
