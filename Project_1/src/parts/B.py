import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from .. import utils
from .. import regression
from .A import create_data


def fit_ridge(x, y, d, l):
    X = utils.poly_features(x, d, intercept=True)
    return regression.ridge(X, y, l)


# note: this is very similar to part A
def heatmap():
    n = 1e3
    x, y = create_data(n)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    degrees = np.arange(1, 15)
    lambdas = np.logspace(-5, 0, 40)

    ds, ls = np.meshgrid(degrees, lambdas)

    def mse_and_r2(d, l):
        theta = fit_ridge(x_train, y_train, d, l)

        X_test = utils.poly_features(x_test, d, intercept=True)

        pred = X_test @ theta

        mse = utils.MSE(y_test, pred)
        r2 = utils.Rsqd(y_test, pred)

        return mse, r2

    # vectorization handles the for loops for us
    mse_and_r2_vec = np.vectorize(mse_and_r2, signature="(),()->(),()")

    mse, r2 = mse_and_r2_vec(ds, ls)

    # Filter out points with r^2 < 0
    mse[r2 < 0] = np.nan
    r2[r2 < 0] = np.nan

    # ------------ Plot it all -------------
    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(8, 3))

    # MSE
    ax = axs[0]
    cmap_mse = plt.colormaps["viridis"]
    cmap_mse.set_bad(color="dimgray")

    cf = ax.pcolormesh(
        ds,
        ls,
        mse,
        shading="nearest",
        cmap=cmap_mse,
        vmin=0,
        vmax=0.1,
    )

    fig.colorbar(cf)

    ax.set_yscale("log")
    ax.set_ylabel(r"$\lambda$")
    ax.set_xlabel("Polynomial degree")
    ax.set_title("MSE")

    # R squared
    ax = axs[1]
    cmap_r2 = plt.colormaps["plasma"]
    cmap_r2.set_bad(color="dimgray")

    cf = ax.pcolormesh(
        ds,
        ls,
        r2,
        shading="nearest",
        cmap=cmap_r2,
        vmin=0,
        vmax=1,
    )

    fig.colorbar(cf)

    ax.set_title("$R^2$")
    ax.set_xlabel("Polynomial degree")

    # title & save
    fig.suptitle(f"Polynomial Ridge Regression on $n={x_train.shape[0]}$ points")
    fig.savefig(os.path.join(utils.FIGURES_URL, "b_heatmap"))
    plt.close()


def main():
    heatmap()


if __name__ == "__main__":
    main()
