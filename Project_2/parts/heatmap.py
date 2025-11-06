import os
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from src import utils, costs
from src.FFNN import FFNN

x, y = utils.generate_regression_data(N=10000, dim=1, noise_std=0.05)
# quick scaling
offset = np.min(y)
scale = 1 / np.max(y - offset)
y = (y - offset) * scale


# This function is broken out so that it
# may be parallelized
def _train(param_tuple):

    nnodes, nlay = param_tuple

    layer_output_sizes = np.zeros(nlay + 1) + nnodes
    layer_output_sizes[-1] = y.shape[0]

    activation_funcs = []
    activation_ders = []
    for _ in layer_output_sizes:
        activation_funcs.append(costs.LeakyReLU)
        activation_ders.append(costs.LeakyReLU_der)

    nn = FFNN(
        network_input_size=x.shape[0],
        layer_output_sizes=layer_output_sizes,
        activation_funcs=activation_funcs,
        activation_ders=activation_ders,
        cost_fun=costs.mse,
        cost_der=costs.mse_der,
        eta=0.2,
        # eta=1e-4,
        batch_size=15,
        regularization_der=None,
        # descent_method="adam",
        # decay_rate=(0.99, 0.99),
    )

    nn.train(x, y, n_iter=3e5)

    print(f"L: {nlay}  N: {nnodes}")

    return nn


def main():
    num_layers = [1, 2, 3]
    num_nodes = [5, 10, 15, 25, 40, 65]

    N, L = np.meshgrid(num_nodes, num_layers)

    combs = zip(N.ravel(), L.ravel())

    # Train networks in parallel
    with Pool(8) as p:
        nns = list(p.map(_train, combs))

    # make test data
    x_test, y_test = utils.generate_regression_data(N=100, dim=1, noise_std=0.05)
    y_test -= offset
    y_test *= scale

    # Calculate mses
    train_mses = np.zeros_like(L, dtype=float)
    test_mses = np.zeros_like(L, dtype=float)

    fig, axs = plt.subplots(
        ncols=len(num_nodes),
        nrows=len(num_layers),
        figsize=(2 * len(num_nodes), 2 * len(num_layers)),
    )

    # ca = np.array(list(combs))
    for j, nlay in enumerate(num_layers):
        for i, nnodes in enumerate(num_nodes):
            idx = j * len(num_nodes) + i

            nn = nns[idx]
            y_pred = nn(x_test)

            axs[j][i].scatter(x_test[0], y_test[0], s=0.5, c="k")
            axs[j][i].plot(x_test[0], y_pred[0], c="orange", lw=2)
            axs[j][i].set_title(f"{nlay} Layers, {nnodes} Nodes")

            test_mses[j][i] = costs.mse(y_pred[0], y_test[0])
            train_mses[j][i] = costs.mse(nn(x)[0], y[0])

            print(f"mses[{j}][{i}]", test_mses[j][i])

    fig.savefig(os.path.join(utils.FIGURES_URL, "ex"))

    plt.close(fig)

    fig, axs = plt.subplots(nrows=2, figsize=(utils.APS_COL_W, 1.6 * utils.APS_COL_W))

    cmap = plt.colormaps["viridis"]
    cmap.set_bad("grey")

    train_mses[train_mses > 0.01] = np.nan
    test_mses[test_mses > 0.01] = np.nan

    norm = colors.Normalize(
        vmin=min(np.nanmin(train_mses), np.nanmin(test_mses)),
        vmax=max(np.nanmax(train_mses), np.nanmax(test_mses)),
    )

    axs[0].imshow(
        train_mses,
        cmap=cmap,
        norm=norm,
    )
    im = axs[1].imshow(
        test_mses,
        cmap=cmap,
        norm=norm,
    )

    axs[0].set_title("Training data")
    axs[1].set_title("Test data")
    fig.suptitle("NN Performance")

    for ax in axs:
        ax.set_xlabel("Number of hidden layers")
        ax.set_ylabel("Nodes per hidden layer")
        ax.set_yticks(range(len(num_layers)), labels=num_layers)
        ax.set_xticks(range(len(num_nodes)), labels=num_nodes)

    fig.colorbar(im, label="MSE", location="bottom")

    fig.savefig(os.path.join(utils.FIGURES_URL, "heatmap"))
    plt.close(fig)
