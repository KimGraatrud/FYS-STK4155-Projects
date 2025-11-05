import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from multiprocessing import Pool
from src import utils, costs
from src.FFNN import FFNN


def relu_comp():
    N = 10000
    x, y = utils.generate_regression_data(N=N, dim=1, noise_std=0.05)

    # quick scaling
    y -= np.min(y)
    y /= np.max(y)

    nn = FFNN(
        network_input_size=x.shape[0],
        layer_output_sizes=[4, y.shape[0]],
        activation_funcs=[costs.ReLU, costs.ReLU],
        activation_ders=[costs.ReLU_der, costs.ReLU_der],
        cost_fun=costs.mse,
        cost_der=costs.mse_der,
        eta=10.0,
        batch_size=100,
    )
    nn_leaky = FFNN(
        network_input_size=x.shape[0],
        layer_output_sizes=[4, y.shape[0]],
        activation_funcs=[costs.LeakyReLU, costs.LeakyReLU],
        activation_ders=[costs.LeakyReLU_der, costs.LeakyReLU_der],
        cost_fun=costs.mse,
        cost_der=costs.mse_der,
        eta=10.0,
        batch_size=100,
    )

    # instead of splitting into train & test, right now it's
    # easier to just create new data for testing
    x_test, y_test = utils.generate_regression_data(N=100, dim=1, noise_std=0.05)

    # quick scaling
    y_test -= np.min(y_test)
    y_test /= np.max(y_test)

    def callback(i):
        if i % 2000 == 0:
            print(
                i,
                f"MSE: {costs.mse(nn(x_test)[0], y_test[0]):.4f}",
                f"MSE_leaky: {costs.mse(nn_leaky(x_test)[0], y_test[0]):.4f}",
            )

    nn.train(x, y, n_iter=1e5, callback=callback)
    nn_leaky.train(x, y, n_iter=1e5, callback=callback)

    y_pred = nn(x_test)
    y_pred_leaky = nn_leaky(x_test)
    c = "darkslateblue"
    c_leaky = "lightcoral"

    fig = plt.figure(figsize=(utils.APS_COL_W, 1.1 * utils.APS_COL_W))

    gs = fig.add_gridspec(2, 4, height_ratios=[4, 1])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(x_test[0], y_test[0], s=1, c="k", label="Test data")
    ax1.plot(x_test[0], y_pred[0], c=c, lw=1, label="ReLU")
    ax1.plot(x_test[0], y_pred_leaky[0], c=c_leaky, lw=1, label="Leaky ReLU")

    ax1.legend()
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$ (norm.-ed)")
    ax1.set_title("Regression with one hidden layer")

    # ax1.set_title('Test')

    # plot node values
    layer_inps, _ = nn._feed_forward_saver(x_test)
    layer_inps_leaky, _ = nn_leaky._feed_forward_saver(x_test)

    axs = []
    for i in range(len(layer_inps[1])):
        a1 = layer_inps[1][i]
        a2 = layer_inps_leaky[1][i]
        if i == 0:
            axs.append(fig.add_subplot(gs[1, i]))
            axs[i].set_ylabel("$a(z)$")
        else:
            axs.append(fig.add_subplot(gs[1, i], sharey=axs[0]))
            axs[i].get_yaxis().set_visible(False)

        axs[i].plot(x_test[0], a1, c=c)
        axs[i].plot(x_test[0], a2, c=c_leaky)
        axs[i].set_xlabel("$x$")
        axs[i].set_title(f"$a_{i}$")

    axs[0].set_ylim(-1, 2)

    fig.savefig(os.path.join(utils.FIGURES_URL, "b_relu"))


x, y = utils.generate_regression_data(N=10000, dim=1, noise_std=0.05)
# quick scaling
offset = np.min(y)
scale = 1 / np.max(y - offset)
y = (y - offset) * scale


# Used in heatmap
def train(param_tuple):

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


def heatmap():
    num_layers = [
        # 0,
        1,
        2,
        3,
    ]
    num_nodes = [
        5,
        10,
        15,
        25,
        40,
        65,
    ]

    N, L = np.meshgrid(num_nodes, num_layers)

    combs = zip(N.ravel(), L.ravel())

    # Train networks in parallel
    with Pool(8) as p:
        nns = list(p.map(train, combs))

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

    fig.savefig(os.path.join(utils.FIGURES_URL, "b_heatmap"))
    plt.close(fig)


# def overfit():


def test():
    dim = 1
    N = 10000
    x, y = utils.generate_regression_data(N=N, dim=dim, noise_std=0.05)

    # # quick scaling
    # y -= np.min(y)
    # y /= np.max(y)

    der = None
    # der = lambda X: costs.L1_der(X, lam=1e-5)
    # der = lambda X: costs.L2_der(X, lam=1e-4)

    nn = FFNN(
        network_input_size=x.shape[0],
        layer_output_sizes=[
            25,
            25,
            y.shape[0],
        ],
        activation_funcs=[
            # costs.sigmoid,
            # costs.LeakyReLU,
            costs.LeakyReLU,
            costs.LeakyReLU,
            costs.LeakyReLU,
            # costs.ReLU,
        ],
        activation_ders=[
            # costs.sigmoid_der,
            # costs.LeakyReLU_der,
            costs.LeakyReLU_der,
            costs.LeakyReLU_der,
            costs.LeakyReLU_der,
            # costs.ReLU_der,
        ],
        cost_fun=costs.mse,
        cost_der=costs.mse_der,
        eta=0.2,
        batch_size=15,
        regularization_der=der,
        # descent_method="adam",
        # decay_rate=(0.99, 0.99),
    )

    # instead of splitting into train & test, right now it's
    # easier to just create new data for testing
    x_test, y_test = utils.generate_regression_data(N=100, dim=dim, noise_std=0.05)

    # # quick scaling
    # y_test -= np.min(y_test)
    # y_test /= np.max(y_test)

    def callback(i):
        if i % 1000 == 0:
            print(
                i,
                f"MSE train {costs.mse(nn(x)[0], y[0]):.4f}",
                f"test {costs.mse(nn(x_test)[0], y_test[0]):.4f}",
            )

    # try:
    nn.train(x, y, n_iter=1e5, callback=callback)
    # except KeyboardInterrupt:
    #     print("\nplotting anyway")

    y_pred = nn(x_test)

    if dim == 2:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(x_test[0], x_test[1], y_test[0], s=1, alpha=1)
        ax.scatter(x_test[0], x_test[1], y_pred[0], s=1, alpha=1)
        plt.show()
    else:
        fig, ax = plt.subplots()
        ax.scatter(x_test[0], y_test[0], s=0.5)
        ax.scatter(x_test[0], y_pred[0])
        ax.legend()
        fig.savefig(os.path.join(utils.FIGURES_URL, "test"))


def main():
    # relu_comp()
    # heatmap()
    test()


if __name__ == "__main__":
    main()
