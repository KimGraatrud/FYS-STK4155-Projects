import os
import matplotlib.pyplot as plt
import numpy as np
from .A import create_data
from .. import ml, utils, regression


def compare_all():
    degree = 5
    n_max = 2e4  # iterations

    N = 1e2
    x, y = create_data(N)

    gd = ml.GD(
        full_output=True, eta=1e-1, atol=None, n_iterations=n_max
    )  # atol=None forces it to run all iterations

    X = utils.poly_features(x, degree, intercept=True)

    fig, ax = plt.subplots()

    # calculate the MSEs from a record of parameters
    def MSE_from_record(record):
        preds = record @ X.T
        return np.array([utils.MSE(y, p) for p in preds])

    # OLS
    _, stats = gd.Grad(X, y)
    MSE_ols = MSE_from_record(stats["record"])
    it = np.arange(MSE_ols.shape[0])
    ax.plot(it, MSE_ols, label="OLS")

    # Ridge
    gd.lamb = 1e-2
    _, stats = gd.Grad(X, y)
    MSE_ridge = MSE_from_record(stats["record"])
    ax.plot(it, MSE_ridge, label="Ridge")

    # OLS w/ momentum
    gd.lamb = 0
    gd.mass = 0.2
    _, stats = gd.Grad(X, y)
    MSE_mntm = MSE_from_record(stats["record"])
    ax.plot(it, MSE_mntm, label="OLS w/ momentum")

    # ADAGrad
    gd.mass = 0
    delta = 1e-7
    _, stats = gd.AdaGrad(X, y, delta=delta)
    MSE_ada = MSE_from_record(stats["record"])
    ax.plot(it, MSE_ada, label="ADAGrad")

    # RMS
    decay = 0.99
    _, stats = gd.RMSGrad(X, y, delta, decay)
    MSE_rms = MSE_from_record(stats["record"])
    ax.plot(it, MSE_rms, label="RMSGrad")

    # ADAM
    decay_1 = 0.9
    decay_2 = 0.9
    delta = 1e-8
    _, stats = gd.ADAM(X, y, delta, decay_1, decay_2)
    MSE_adam = MSE_from_record(stats["record"])
    ax.plot(it, MSE_adam, label="ADAM")

    ax.set_xscale("log")
    # ax.set_yscale("symlog", linthresh=0.1)
    # ax.set_xscale("symlog", linthresh=100)

    # optimal
    theta = regression.OLS(X, y)
    MSE_best = utils.MSE(y, X @ theta)
    ax.axhline(MSE_best, ls="--", label="Optimal (OLS analytical soln.)")

    ax.set_ybound(MSE_best - 0.005, 0.15)
    ax.set_xbound(1, n_max)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training MSE")
    ax.set_title("MSE during descent, various methods")

    ax.legend()

    fig.savefig(os.path.join(utils.FIGURES_URL, "d"))
    plt.close()


def main():
    compare_all()


if __name__ == "__main__":
    main()
