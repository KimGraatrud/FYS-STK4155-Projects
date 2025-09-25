import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from .. import utils
from .. import regression

rng = np.random.default_rng(seed=1000)


def create_data(n):
    # create data
    x = np.linspace(-1, 1, np.int32(n))
    y_base = utils.runge(x)
    noise = 0.05 * rng.normal(size=x.shape[0])
    y = y_base + noise

    return x, y


def fit_OLS(x, y, d):
    X = utils.poly_features(x, d, intercept=True)
    return regression.OLS(X, y)


def heatmap():
    n = 1e4
    x, y = create_data(n)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    degrees = np.arange(1, 15)
    fracs = np.logspace(-3, 0, 40)
    intervals = np.int32(np.round(1 / fracs))  # use every ith datapoint

    ds, ints = np.meshgrid(degrees, intervals)

    # Function that computes the MSE and r^2 for one degree and interval
    def mse_and_r2(d, i):
        # find optimal parameters
        x_sampled = x_train[::i]
        y_sampled = y_train[::i]

        theta = fit_OLS(x_sampled, y_sampled, d)

        # predict test vales
        X_test = utils.poly_features(x_test, d, intercept=True)

        pred = X_test @ theta

        mse = utils.MSE(y_test, pred)
        r2 = utils.Rsqd(y_test, pred)

        return mse, r2

    # vectorization handles the for loops for us
    mse_and_r2_vec = np.vectorize(mse_and_r2, signature="(),()->(),()")

    mse, r2 = mse_and_r2_vec(ds, ints)

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
        n / ints,
        mse,
        shading="nearest",
        cmap=cmap_mse,
        vmin=0,
        vmax=0.1,
    )

    fig.colorbar(cf)

    ax.set_yscale("log")
    ax.set_ylabel("Number of points")
    ax.set_xlabel("Polynomial degree")
    ax.set_title("MSE")

    # R squared
    ax = axs[1]
    cmap_r2 = plt.colormaps["plasma"]
    cmap_r2.set_bad(color="dimgray")

    cf = ax.pcolormesh(
        ds,
        n / ints,
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
    fig.suptitle("OLS polynomial fitting")
    fig.savefig(os.path.join(utils.FIGURES_URL, "a_heatmap"))
    plt.close()


def example():
    n = 1e4
    x, y = create_data(n)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    i = 80  # only train on every 80th point
    x_sample = x_train[::i]
    y_sample = y_train[::i]

    # plot the underlying data
    c = "k"
    ax.scatter(x, y, s=0.1, alpha=0.1, c=c)
    ax.scatter(x_sample, y_sample, s=1.2, c=c, label="Training points")

    n_deg = 12  # number of degrees to plot

    # colormaps for coloring the lines
    cmap = plt.colormaps["cividis"].reversed()
    norm = mpl.colors.Normalize(vmin=1, vmax=n_deg)

    # fit to various degrees
    degrees = np.arange(1, n_deg + 1)
    for d in degrees:
        theta = fit_OLS(x_sample, y_sample, d)

        X = utils.poly_features(x, d, intercept=True)

        ax.plot(x, X @ theta, c=cmap(norm(d)), lw=1)

    fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Polynomial degree"
    )

    ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    fig.suptitle("Polynomial OLS fits")
    fig.savefig(os.path.join(utils.FIGURES_URL, "a_example"))


def main():
    heatmap()
    example()


if __name__ == "__main__":
    main()
