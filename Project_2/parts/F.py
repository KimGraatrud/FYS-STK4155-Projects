import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.FFNN import FFNN
from src.ClassifierNN import ClassifierNN
from src import utils


def main():
    # Fetch the MNIST dataset
    mnist = utils.load_openml_dataset()

    # Extract data (features) and target (labels)
    X = mnist["data"]
    y = mnist["target"]

    # Split into training and testing
    X_train, X_test, y_train, y_test = utils.train_test_split(X, y)

    # Format data for our network
    X_train = X_train.T / 255.0
    X_test = X_test.T / 255.0
    y_train = utils.onehot(np.int32(y_train))
    y_test = utils.onehot(np.int32(y_test))

    nn = ClassifierNN(
        network_input_size=X_train.shape[0],
        classes=y_train.shape[0],
        layer_output_sizes=[
            # 10,
            # 15,
        ],
        activation_funcs=[
            # utils.ReLU,
            # utils.ReLU,
        ],
        activation_ders=[
            # utils.ReLU_der,
            # utils.ReLU_der,
        ],
        eta=1e-3,
        batch_size=150,
    )

    # quick function for reporting
    def train_callback(i):
        if i % 500 == 0:
            train = utils.accuracy(nn(X_train), y_train)
            test = utils.accuracy(nn(X_test), y_test)
            print(f"train: {train:.4f} test: {test:.4f} iter: {i}")

            # W, b = nn.layers[0]
            # fig, axs = plt.subplots(nrows=W.shape[1], figsize=(4, 16))
            # for n in range(W.shape[1]):
            #     ax = axs[n]
            #     ax.imshow(W[:, n].reshape(28, 28))

            # fig.savefig(os.path.join(utils.FIGURES_URL, f"layer_{i}"))

            # plt.close(fig)

    nn.train(X_train, y_train, n_iter=2001, callback=train_callback)

    pred = nn(X_test)
    norm = mpl.colors.Normalize(vmin=-0.75, vmax=0.75)
    W, b = nn.layers[0]
    for i in range(7):
        fig, axs = plt.subplots(nrows=W.shape[1], ncols=3, figsize=(8, 16))

        inp = X_test[:, i]
        out = pred[:, i]

        axs[0][0].imshow(inp.reshape(28, 28), norm=norm)

        for n in range(W.shape[1]):
            axs[n][1].imshow(W[:, n].reshape(28, 28), norm=norm)
            result = W[:, n] * inp
            # result = result / np.mean(result) - 1
            axs[n][2].imshow(result.reshape(28, 28))
            axs[n][2].set_ylabel(f"{out[n]:.2f}")
            if n != 0:
                fig.delaxes(axs[n][0])

        fig.savefig(os.path.join(utils.FIGURES_URL, f"result_{i}"))
        plt.close(fig)


if __name__ == "__main__":
    main()
