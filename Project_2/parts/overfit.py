import os
import numpy as np
import matplotlib.pyplot as plt
from src import utils, costs
from src.FFNN import FFNN


def main():
    N = 100
    x, y = utils.generate_regression_data(N=N, dim=1, noise_std=0.05)
    y -= np.min(y)
    y /= np.max(y)

    networks = [
        [5],
        [10, 10],
        [50, 50],
    ]

    rates = [
        1e-4,
        1e-2,
        1e-1,
    ]

    nns = []
    for i, network in enumerate(networks):
        nns.append(
            FFNN(
                network_input_size=x.shape[0],
                layer_output_sizes=[
                    *network,
                    y.shape[0],
                ],
                activation_funcs=[
                    costs.LeakyReLU,
                    costs.LeakyReLU,
                    costs.LeakyReLU,
                ],
                activation_ders=[
                    costs.LeakyReLU_der,
                    costs.LeakyReLU_der,
                    costs.LeakyReLU_der,
                ],
                cost_fun=costs.mse,
                cost_der=costs.mse_der,
                eta=rates[i],
                descent_method="adam",
                decay_rate=(0.99, 0.99),
            )
        )

    try:
        for nn in nns:
            nn.train(x, y, n_iter=3e4)
    except KeyboardInterrupt:
        pass

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))
    ax.scatter(x[0], y[0], s=0.7, c="k")

    colors = ["lightblue", "cornflowerblue", "navy"]
    labels = ["$5$", "$10, 10$", "$50, 50$"]
    for i, nn in enumerate(nns):
        ax.plot(x[0], nn(x)[0], label=labels[i], c=colors[i])

    ax.legend()
    ax.set_title("Overfitting demonstration")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\hat{y}$")

    fig.savefig(os.path.join(utils.FIGURES_URL, "overfit"))
    plt.close(fig)
