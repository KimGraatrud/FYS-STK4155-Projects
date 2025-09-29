import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from .A import create_data
from .. import ml, utils, regression


# calculate the MSEs from a record of parameters
def MSE_from_record(X, y, record):
    preds = record @ X.T
    return np.array([utils.MSE(y, p) for p in preds])


# Not worth it for
# - Adagrad: the small constant is just for numerical stability


def setup_comparison():
    degree = 5
    n_max = 2e4  # iterations

    N = 1e2
    x, y = create_data(N)
    X = utils.poly_features(x, degree, intercept=True)

    gd = ml.GD(full_output=True, eta=1e-1, atol=None, n_iterations=n_max)

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    # optimal
    theta = regression.OLS(X, y)
    MSE_best = utils.MSE(y, X @ theta)
    ax.axhline(
        MSE_best,
        ls="--",
        label="OLS analytical soln.",
        c="k",
    )

    # OLS
    _, stats = gd.Grad(X, y)
    MSE_ols = MSE_from_record(X, y, stats["record"])
    ax.plot(MSE_ols, label="OLS")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training MSE")

    ax.set_xscale("log")

    ax.set_ylim(MSE_best - 0.005, 0.13)
    ax.set_xlim(1, n_max)

    return fig, ax, X, y, gd


def compare_rms():
    fig, ax, X, y, gd = setup_comparison()

    cmap = plt.colormaps["Reds"]
    norm = mpl.colors.Normalize(vmin=0.5, vmax=1.2)

    # RMSGrad
    decay = np.array([0.7, 0.9, 0.99])
    for d in decay:
        _, stats = gd.RMSGrad(X, y, 1e-7, d)
        MSE = MSE_from_record(X, y, stats["record"])
        label = r"$\rho = " + f"{d}$"
        ax.plot(MSE, label=label, c=cmap(norm(d)))

    ax.set_title("RMSProp")
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.set_figheight(0.9 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "d_rms"))
    plt.close()


def compare_adam():
    fig, ax, X, y, gd = setup_comparison()

    cmap = plt.colormaps["Purples"]
    norm = mpl.colors.Normalize(vmin=0.5, vmax=1.2)

    # ADAM
    decay_1 = np.array([0.7, 0.9, 0.99])
    d2 = 0.99
    for d1 in decay_1:
        _, stats = gd.ADAM(X, y, 1e-7, d1, d2)
        MSE = MSE_from_record(X, y, stats["record"])
        label = r"$\rho_1 = " + f"{d1}" + r"$, $\rho_2 = " + f"{d2}$"
        ax.plot(MSE, label=label, c=cmap(norm(d1)))

    d1 = 0.9
    decay_2 = np.array([0.7, 0.999])
    lss = ["--", "-."]
    for d2, ls in zip(decay_2, lss):
        _, stats = gd.ADAM(X, y, 1e-7, d1, d2)
        MSE = MSE_from_record(X, y, stats["record"])
        label = r"$\rho_1 = " + f"{d1}" + r"$, $\rho_2 = " + f"{d2}$"
        ax.plot(MSE, label=label, ls=ls, c=cmap(norm(d1)))

    ax.set_title("ADAM")
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.set_figheight(0.97 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "d_adam"))
    plt.close()


def compare_momentum():
    fig, ax, X, y, gd = setup_comparison()

    masses = np.array([0.1, 0.5, 0.8, 0.92])

    cmap = plt.colormaps["Oranges"]
    norm = mpl.colors.Normalize(vmin=-0.5, vmax=1.5)

    for m in masses:
        # OLS w/ momentum
        gd.lamb = 0
        gd.mass = m
        _, stats = gd.Grad(X, y)
        MSE_mntm = MSE_from_record(X, y, stats["record"])
        ax.plot(MSE_mntm, label=f"mass = {m}", c=cmap(norm(m)))

    ax.set_title("OLS + Momentum")
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.set_figheight(0.9 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "d_mass"))
    plt.close()


def compare_lasso():
    fig, ax, X, y, gd = setup_comparison()

    l_min = -5
    l_max = -1
    lambs = np.logspace(l_min, l_max, 5)

    cmap = plt.colormaps["YlGn"]
    norm = mpl.colors.Normalize(vmin=l_min - 3, vmax=l_max)

    # Ridge for comparison
    gd.lamb = 1e-2
    _, stats = gd.Grad(X, y)
    MSE_ridge = MSE_from_record(X, y, stats["record"])
    ax.plot(MSE_ridge, label=r"Ridge, $\lambda=10^" + f"{{{np.log10(gd.lamb):.0f}}}$")

    # Lasso
    for l in lambs:
        gd.lamb = l
        _, stats = gd.Lasso(X, y)
        MSE_lasso = MSE_from_record(X, y, stats["record"])
        label = r"$\lambda=" + f"10^{{{np.log10(l):.0f}}}$"
        ax.plot(MSE_lasso, label=label, c=cmap(norm(np.log10(l))))

    ax.set_title("Lasso")
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.set_figheight(0.97 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "d_lasso"))
    plt.close()


def compare_all():
    fig, ax, X, y, gd = setup_comparison()

    # OLS w/ momentum
    gd.lamb = 0
    gd.mass = 0.2
    _, stats = gd.Grad(X, y)
    MSE_mntm = MSE_from_record(X, y, stats["record"])
    ax.plot(MSE_mntm, label="OLS + momentum")

    # Ridge
    gd.lamb = 1e-2
    gd.mass = 0
    _, stats = gd.Grad(X, y)
    MSE_ridge = MSE_from_record(X, y, stats["record"])
    ax.plot(MSE_ridge, label="Ridge")

    # RMS
    gd.lamb = 0
    decay = 0.99
    delta = 1e-7
    _, stats = gd.RMSGrad(X, y, delta, decay)
    MSE_rms = MSE_from_record(X, y, stats["record"])
    # break the RMS plot into stable and unstable
    it = np.arange(MSE_rms.shape[0])
    stable = it < 1.5e2
    rms_color = "#d43b38"
    ax.plot(it[stable], MSE_rms[stable], label="RMSProp", c=rms_color, zorder=0)
    ax.plot(it[~stable], MSE_rms[~stable], c=rms_color, lw=0.3, zorder=0)

    # ADAGrad
    gd.mass = 0
    _, stats = gd.AdaGrad(X, y, delta=delta)
    MSE_ada = MSE_from_record(X, y, stats["record"])
    ax.plot(MSE_ada, label="ADAGrad", c="peru")

    # ADAM
    decay_1 = 0.9
    decay_2 = 0.9
    delta = 1e-8
    _, stats = gd.ADAM(X, y, delta, decay_1, decay_2)
    MSE_adam = MSE_from_record(X, y, stats["record"])
    ax.plot(MSE_adam, label="ADAM", c="mediumpurple")

    # Lasso
    _, stats = gd.Lasso(X, y)
    MSE_lasso = MSE_from_record(X, y, stats["record"])
    ax.plot(MSE_lasso, label="Lasso", c="yellowgreen")

    ax.set_title("MSE during descent, various methods")
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.set_figheight(0.97 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "d"))
    plt.close()


def main():
    compare_momentum()
    compare_all()
    compare_adam()
    compare_rms()
    compare_lasso()


if __name__ == "__main__":
    main()
