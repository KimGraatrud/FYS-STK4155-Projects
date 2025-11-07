import os
import numpy as np
import matplotlib.pyplot as plt
from src import utils, costs
from src.FFNN import FFNN


def main():
    N = 10000
    x, y = utils.generate_regression_data(N=N, dim=1, noise_std=0.05)

    # quick scaling
    y -= np.min(y)
    y /= np.max(y)

    avns = [costs.ReLU, costs.LeakyReLU, costs.sigmoid]
    avn_ders = [costs.ReLU_der, costs.LeakyReLU_der, costs.sigmoid_der]

    nns = []
    for a, da in zip(avns, avn_ders):
        nn = FFNN(
            network_input_size=x.shape[0],
            layer_output_sizes=[8, y.shape[0]],
            activation_funcs=[a, costs.one],
            activation_ders=[da, costs.one_der],
            cost_fun=costs.mse,
            cost_der=costs.mse_der,
            eta=1e-3,
            batch_size=32,
            descent_method="adam",
            decay_rate=(0.9, 0.999),
        )

        def callback(i):
            if i % 2000 == 0:
                print(i, f"MSE: {costs.mse(nn(x)[0], y[0]):.4f}")

        nn.train(x, y, n_iter=5e4, callback=callback)

        nns.append(nn)

    fig = plt.figure(figsize=(utils.APS_COL_W, 1.0 * utils.APS_COL_W))

    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[3, 1])

    z = np.linspace(-4, 4, 50)

    ax = fig.add_subplot(gs[0, :])

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    colors = ["royalblue", "skyblue", "sandybrown"]

    idx = utils.rng.integers(N, size=32)
    bx, by = x[0, idx], y[0, idx]
    ax.scatter(bx, by, c="k", s=1.0, alpha=1, label="Ex. Batch")
    for i in range(3):
        ax.plot(x[0], nns[i](x)[0], c=colors[i])

    ax.legend()

    rax1 = fig.add_subplot(gs[1, 0])
    rax2 = fig.add_subplot(gs[1, 1], sharex=rax1, sharey=rax1)
    rax3 = fig.add_subplot(gs[1, 2], sharex=rax1, sharey=rax1)

    rax1.plot(z, costs.ReLU(z), c=colors[0])
    rax1.set_title(r"ReLU")

    rax2.plot(z, costs.LeakyReLU(z), c=colors[1])
    rax2.set_title(r"Leaky ReLU")

    rax3.plot(z, costs.sigmoid(z), c=colors[2])
    rax3.set_title(r"Sigmoid")

    rax1.set_ylim(-0.4, 2)
    rax1.set_xlim(-4, 4)

    rax1.set_ylabel(r"$a(z)$")
    # rax2.set_ylabel(r"$a(z)$")
    # rax3.set_ylabel(r"$a(z)$")

    rax2.get_yaxis().set_visible(False)
    rax3.get_yaxis().set_visible(False)

    for rax in (rax1, rax2, rax3):
        rax.axhline(0, lw=0.2, c="k", alpha=0.7)
        rax.axvline(0, lw=0.2, c="k", alpha=0.7)

    rax1.set_xlabel(r"$z$")
    rax2.set_xlabel(r"$z$")
    rax3.set_xlabel(r"$z$")

    ax.set_title(r"Effect of $a(z)$ choice")

    fig.savefig(os.path.join(utils.FIGURES_URL, "actvns"))
