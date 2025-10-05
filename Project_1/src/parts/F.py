import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from .. import ml, utils, regression
from . import A

# TODO: put this somewhere else?
n = 1e5  # data points
deg = 5
Ms = np.logspace(np.log10(n), 1, 40)

x, y = A.create_data(n=n)
X = utils.poly_features(x, deg)


def sgd_generate():
    """
    Create figure tk.tk
    """
    x, y = A.create_data(n=n)
    X = utils.poly_features(x, deg)

    # gd = ml.GD(X, y,n_iterations=1e4, eta=1e-1)
    # theta = gd.Grad()
    # theta_opt = regression.OLS(X, y)
    # mse_grad = utils.MSE(y, X @ theta)
    # print("mse_grad", mse_grad)
    # mse_opt = utils.MSE(y, X @ theta_opt)
    # print("mse_opt", mse_opt)

    mses = {
        "ols": [],
        "mass": [],
        "ridge": [],
        "rms": [],
        "ada": [],
        "adam": [],
        "lasso": [],
    }

    for M in Ms:
        print("M", M)
        # TODO: run this with the usual n and eta
        gd = ml.GD(X, y, M=M, n_iterations=1e4, eta=1e-1)
        # this is similar to compare_all in part D

        mses["ols"].append(utils.MSE(y, X @ gd.Grad()))

        gd.mass = 0.2
        mses["mass"].append(utils.MSE(y, X @ gd.Grad()))
        gd.mass = 0

        gd.lamb = 1e-2
        mses["ridge"].append(utils.MSE(y, X @ gd.Grad()))
        gd.lamb = 0

        decay = 0.99
        delta = 1e-7
        mses["rms"].append(utils.MSE(y, X @ gd.RMSGrad(delta, decay)))

        mses["ada"].append(utils.MSE(y, X @ gd.AdaGrad(delta)))

        decay_1 = 0.9
        decay_2 = 0.9
        delta = 1e-8
        mses["adam"].append(utils.MSE(y, X @ gd.ADAM(delta, decay_1, decay_2)))

        mses["lasso"].append(utils.MSE(y, X @ gd.Lasso()))

    path = os.path.join(utils.DATA_URL, "d_sgd.pkl")

    # Save the data so we don't have to recalculate every time we plot
    with open(path, mode="+wb") as f:
        pkl = pickle.Pickler(f)
        pkl.dump(mses)


def sgd_plot():
    path = os.path.join(utils.DATA_URL, "d_sgd.pkl")
    with open(path, mode="rb") as f:
        pkl = pickle.Unpickler(f)
        mses = pkl.load()

    labels = {
        "ols": "OLS",
        "mass": "OLS + momentum",
        "ridge": "Ridge",
        "rms": "RMSProp",
        "ada": "ADAGrad",
        "adam": "ADAM",
        "lasso": "Lasso",
    }

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.97 * utils.APS_COL_W))

    for key in labels.keys():
        mse = mses[key]
        ax.plot(Ms, mse, label=labels[key], c=utils.colors[key])

    ax.set_xlabel("Batch size $M$")
    ax.set_ylabel("MSE")
    ax.set_xscale("log")
    ax.set_title("Effect of Batch Size on Final MSE")

    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.savefig(os.path.join(utils.FIGURES_URL, "f"))


#     raise
#     mses = np.array(mses)

#     # also do analytical
#     theta_opt = regression.OLS(X, y)
#     mse_opt = utils.MSE(y, X @ theta_opt)

#     ax.plot(Ms, mses)

#     ax.axhline(mse_opt, c="k", ls="--")


def main():
    sgd_generate()
    sgd_plot()


if __name__ == "__main__":
    main()
