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


def heatmap():
    """
    Creates Figure tk.tk
    """
    n = 1e3
    x, y = create_data(n)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=utils.RANDOM_SEED
    )

    degrees = np.arange(1, 15)
    etas = np.logspace(-6, -1, 15)  # learning rates

    ds, es = np.meshgrid(degrees, etas)

    mse_diffs = np.zeros_like(ds, dtype=np.float64)

    n_iterations = 1e3  # number of iterations to run each grad. descent for

    for i, d in enumerate(degrees):
        X_test = utils.poly_features(x_test, d, intercept=True)

        theta_ridge = fit_ridge(x_train, y_train, d, 0)  # OLS
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
    fig.colorbar(cf, label=r"$\mathrm{MSE}_{\nabla} - \mathrm{MSE}_\mathrm{OLS}$")

    ax.set_yscale("log")

    ax.set_ylabel(r"learning rate $\eta$")
    ax.set_xlabel(r"polynomial degree")
    ax.set_title(
        r"Diff. in MSE between $\nabla$-descent and OLS, "
        + f"{n_iterations:.0f} iterations",
    )

    fig.savefig(os.path.join(utils.FIGURES_URL, "c_heatmap"))
    plt.close()


def eta_n_relationship():
    # This should really be used to compare models. Useless on its own

    degree = 4  # arbitrary
    max_n = 1e6  # maximum number of iterations
    atol = 1e-3
    # atol = 1e-8

    n_data = 1e2
    x, y = create_data(n_data)

    etas = np.logspace(-3, -0.1, 20)  # learning rates
    ns = np.zeros_like(etas)  # number of iterations before stopping

    gd = ml.GD(n_iterations=max_n, atol=atol, full_output=True)
    # gd = ml.GD(n_iterations=max_n, atol=atol, mass=1)

    X = utils.poly_features(x, degree, intercept=True)

    for i, eta in enumerate(etas):
        gd.eta = eta
        _, stats = gd.Grad(X, y)
        ns[i] = stats["n"]

    fig, ax = plt.subplots()
    ax.plot(etas, ns)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Learning rate $\eta$")
    ax.set_ylabel(r"Iterations before stopping")

    ax.set_title(
        r"Iterations vs. Learning Rate for $\nabla$-Descent of Degree 4 Polynomial"
    )

    fig.savefig(os.path.join(utils.FIGURES_URL, "c_eta_n_rel"))
    plt.close()


def main():
    heatmap()
    eta_n_relationship()


if __name__ == "__main__":
    main()
