import os
import numpy as np
import matplotlib.pyplot as plt
from src.parts.A import create_data
from src import utils, ml, regression


def main():
    degree = 5

    N = 1e2
    x, y = create_data(N)
    X = utils.poly_features(x, degree, intercept=True)

    gd = ml.GD(X, y, verbose=False, eta=1e-1, atol=None, n_iterations=1e4)

    mse_opt = utils.MSE(y, X @ regression.OLS(X, y))

    mse = utils.MSE(y, X @ gd.Grad())

    print("Delta mse", mse - mse_opt)
    # n_max = 2e4  # iterations
    # gd = ml.GD(X, y, verbose=True, eta=1e-1, atol=None, n_iterations=n_max)

    # theta, stats = gd.RMSGrad()

    # def MSE_from_record(X, y, record):
    #     preds = record @ X.T
    #     return np.array([utils.MSE(y, p) for p in preds])

    # fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    # mse = MSE_from_record(X, y, stats["record"])

    # i = np.arange(len(mse))

    # filter = (0 < i) & (i < 5e2)

    # ax.plot(i[filter], mse[filter])
    # # ax.plot(i[filter], np.linalg.norm(stats["r"][filter], axis=1), label="r")
    # # ax.plot(i[filter], np.min(stats["r"][filter], axis=1), label="r")
    # # ax.plot(i[filter], np.linalg.norm(stats["grads"][filter], axis=1))
    # ax.plot(i[filter], np.min(stats["sqrt"][filter], axis=1), label="sqrt")
    # # ax.plot(i[filter], np.min(stats["grads"][filter], axis=1), label="grads")
    # # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.legend()
    # fig.savefig(os.path.join(utils.FIGURES_URL, "test"))

    # print(mse)


if __name__ == "__main__":
    main()
