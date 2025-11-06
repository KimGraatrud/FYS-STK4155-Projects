import os
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from src import utils
from src.ClassifierNN import ClassifierNN


def _train(params):
    batch_size = params["batch_size"]
    n_iter = params["n_iter"]
    epoch = params["epoch"]
    X = params["X"]
    y = params["y"]

    nn = ClassifierNN(
        network_input_size=X.shape[0],
        classes=y.shape[0],
        layer_output_sizes=[
            # 8,
            # 15
        ],
        activation_funcs=[
            # costs.ReLU,
            # costs.ReLU,
        ],
        activation_ders=[
            # costs.LeakyReLU_der,
            # costs.ReLU_der,
        ],
        eta=1.0,
        batch_size=batch_size,
        regularization_der=None,
        # descent_method="adam",
        # decay_rate=(0.9, 0.999),
    )

    nn.train(X, y, n_iter=n_iter)

    print(f"{epoch}: {batch_size} for {n_iter} iterations")

    return nn


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

    N = X.shape[0]

    n_sizes = 20
    batch_sizes = np.int32(np.logspace(np.log10(2), np.log10(N / 50), n_sizes))
    # epochs = [1]
    epochs = [1, 2]

    # build input objects
    params = []
    for bs in batch_sizes:
        for epoch in epochs:
            params.append(
                {
                    "batch_size": bs,
                    "epoch": epoch,
                    "n_iter": 10**epoch,
                    # "n_iter": np.int32((N / bs) * epoch),
                    "X": X_train,
                    "y": y_train,
                }
            )

    # Train in parallel
    with Pool(7) as p:
        nns = list(p.map(_train, params))

    scores = {}
    bses = {}
    for nn, ps in zip(nns, params):
        epoch = ps["epoch"]
        if epoch not in scores:
            scores[epoch] = []
        if epoch not in bses:
            bses[epoch] = []

        pred = nn(X_test)

        scores[epoch].append(utils.accuracy(pred, y_test))
        bses[epoch].append(ps["batch_size"])

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.8 * utils.APS_COL_W))

    for epoch in epochs:
        ax.plot(bses[epoch], scores[epoch], label=f"{10**epoch}")

    ax.set_title("Accuracy vs. Batch Size")
    ax.set_xlabel("Batch size")
    ax.set_xscale("log")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(os.path.join(utils.FIGURES_URL, "batches"))
    plt.close(fig)
