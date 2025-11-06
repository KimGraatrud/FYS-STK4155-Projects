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

    fig.savefig(os.path.join(utils.FIGURES_URL, "relu"))
