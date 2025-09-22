import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .A import create_data
from .B import fit_ridge
from .. import utils, ml


def fit_grad(x, y, d, **grad_kwargs):
    X = utils.poly_features(x, d, intercept=True)
    descender = ml.GD(**grad_kwargs)
    return descender.Grad(X, y)


def main():
    n = 1e3
    x, y = create_data(n)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    degrees = np.arange(1, 15)
    etas = np.logspace(-6, -1, 15)  # learning rates

    ds, es = np.meshgrid(degrees, etas)

    mse_diffs = np.zeros_like(ds, dtype=np.float64)

    n_iterations = 1e3

    for i, d in enumerate(degrees):
        X_test = utils.poly_features(x_test, d, intercept=True)

        theta_ridge = fit_ridge(x_train, y_train, d, 0)
        # theta_ridge = fit_ridge(x_train, y_train, d, 1e-3)
        pred_ridge = X_test @ theta_ridge
        mse_ridge = utils.MSE(y_test, pred_ridge)

        for j, eta in enumerate(etas):
            theta_grad = fit_grad(
                x_train,
                y_train,
                d,
                eta=eta,
                n_iterations=n_iterations,
            )
            pred_grad = X_test @ theta_grad
            mse_grad = utils.MSE(y_test, pred_grad)

            mse_diffs[j, i] = mse_grad - mse_ridge
            # < 0 implies ridge is worse, > 0 implies grad is worse

    # --- Plot it! ---
    fig, ax = plt.subplots()

    cf = ax.pcolormesh(
        ds,
        es,
        mse_diffs,
        shading="nearest",
        cmap=plt.colormaps["bone"].reversed(),
    )

    ax.set_yscale("log")

    ax.set_title(
        r"Diff. in MSE between $\nabla$-descent and OLS,"
        + f"{n_iterations:.0f} iterations",
    )
    ax.set_ylabel(r"learning rate $\eta$")
    ax.set_xlabel(r"polynomial degree")

    fig.colorbar(cf, label=r"$\mathrm{MSE}_{\nabla} - \mathrm{MSE}_\mathrm{OLS}$")

    fig.savefig(os.path.join(utils.FIGURES_URL, "c_heatmap"))
    plt.close()

    # pred_b
    # print(theta_ridge - theta_grad)


if __name__ == "__main__":
    main()
