import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from .. import utils, regression, resampling
from .A import create_data


def plot_bootstrapdegree(
    degrees: np.ndarray, MSE: np.ndarray, bias: np.ndarray, variance: np.ndarray
) -> None:
    """
    Simple helper func for plotting bootstrap MSE over degrees.
    """

    # Using our standard figuresize
    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    ax.set_xlabel("Degree")
    ax.set_ylabel(r"MSE | Bias | $\sigma^2$")

    cmap = plt.colormaps["Reds"]
    norm = mpl.colors.Normalize(vmin=0.5, vmax=1.2)

    ax.set_title("Bias-Variance")

    ax.plot(degrees, MSE, label="MSE")
    ax.plot(degrees, bias, label="Bias", linestyle="dashed")
    ax.plot(degrees, variance, label="Variance")

    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.set_figheight(0.9 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "BiasVarianceTradeoff"))
    plt.close()


def test_v_train():
    x, y = create_data(n=1e2)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=0.8,
        random_state=utils.RANDOM_SEED,
    )

    train_mses = []
    test_mses = []
    degs = np.arange(40)
    for d in degs:
        X_train = utils.poly_features(x_train, d=d)
        X_test = utils.poly_features(x_test, d=d)

        theta = regression.OLS(X_train, y_train)

        train_mse = utils.MSE(y_train, X_train @ theta)
        test_mse = utils.MSE(y_test, X_test @ theta)

        train_mses.append(train_mse)
        test_mses.append(test_mse)

    train_mses = np.array(train_mses)
    test_mses = np.array(test_mses)

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    ax.plot(degs, train_mses, label="Training")
    ax.plot(degs, test_mses, label="Testing")

    ax.set_ylabel("MSE")
    ax.set_xlabel("Polynomial degree")
    ax.set_title("MSE vs. Polynomial degree")

    # ax.set_xscale("log")
    ax.set_yscale("log")

    ax.legend()

    fig.savefig(os.path.join(utils.FIGURES_URL, "g_train_test"))


def main():
    test_v_train()
    # n_datapoints = int(1e3)
    # maxdeg       = 18
    # x, y = create_data(n_datapoints)

    # strap = resampling.resampling()

    # n_bootstrap = int(1e2)
    # degrees, MSE, bias, variance = strap.BootstrapOLS(
    #     n_bootstrap, maxdeg, x, y, verbose=True
    # )

    # plot_bootstrapdegree(degrees, MSE, bias, variance)


if __name__ == "__main__":
    main()
