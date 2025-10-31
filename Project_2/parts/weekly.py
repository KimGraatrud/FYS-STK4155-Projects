import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src import utils, costs
from src.FFNN import FFNN


def heatmap():
    N = 10000
    x, y = utils.generate_regression_data(N=N, dim=1, noise_std=0.05)

    # quick scaling
    y -= np.min(y)
    y /= np.max(y)

    x_test, y_test = utils.generate_regression_data(N=100, dim=1, noise_std=0.05)

    # quick scaling
    y_test -= np.min(y_test)
    y_test /= np.max(y_test)

    der = None

    num_layers = [0, 1, 2, 3]
    num_nodes = [5, 10, 25, 50]

    L, N = np.meshgrid(num_layers, num_nodes)

    mses = np.zeros_like(N, dtype=float)

    fig, axs = plt.subplots(
        nrows=len(num_layers), ncols=len(num_layers), figsize=(8, 8)
    )

    for i, nlay in enumerate(num_layers):
        for j, nnodes in enumerate(num_nodes):
            print(f"Hidden layers: {nlay} Num nodes: {nnodes}", end="")
            layer_output_sizes = np.zeros(nlay + 1) + nnodes
            layer_output_sizes[-1] = y.shape[0]

            activation_funcs = []
            activation_ders = []
            for _ in layer_output_sizes:
                activation_funcs.append(costs.sigmoid)
                activation_ders.append(costs.sigmoid_der)

            nn = FFNN(
                network_input_size=x.shape[0],
                layer_output_sizes=layer_output_sizes,
                activation_funcs=activation_funcs,
                activation_ders=activation_ders,
                cost_fun=costs.mse,
                cost_der=costs.mse_der,
                eta=1.2,
                batch_size=50,
                regularization_der=der,
            )

            nn.train(x, y, n_iter=2e5)

            y_pred = nn(x_test)

            axs[i][j].scatter(x_test[0], y_test[0], s=0.5)
            axs[i][j].scatter(x_test[0], y_pred[0])
            axs[i][j].set_title(layer_output_sizes)

            mses[i][j] = costs.mse(y_pred, y_test)
            print("mses[i][j]", mses[i][j])

    fig.savefig(os.path.join(utils.FIGURES_URL, "ex"))

    fig, ax = plt.subplots()

    # norm = mpl.colors.Normalize(vmin=0, vmax=0.1)

    im = ax.imshow(mses)
    ax.set_xticks(range(len(num_layers)), labels=num_layers)
    ax.set_yticks(range(len(num_nodes)), labels=num_nodes)

    ax.set_title("NN Performance")
    ax.set_xlabel("Number of hidden layers")
    ax.set_ylabel("Nodes per hidden layer")

    fig.colorbar(im, label="MSE")

    fig.savefig(os.path.join(utils.FIGURES_URL, "heatmap"))
    plt.close(fig)


def main():
    heatmap()


if __name__ == "__main__":
    main()
