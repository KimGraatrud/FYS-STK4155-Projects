import os
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from src import utils
from src.ClassifierNN import ClassifierNN


def _train(params):
    batch_size = params["batch_size"]
    n_iter = params["n_iter"]
    # epoch = params["epoch"]
    X_train = params["X_train"]
    X_test = params["X_test"]
    y_train = params["y_train"]
    y_test = params["y_test"]

    nn = ClassifierNN(
        network_input_size=X_train.shape[0],
        classes=y_train.shape[0],
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
        eta=1e-3,
        batch_size=batch_size,
        regularization_der=None,
        descent_method="adam",
        decay_rate=(0.9, 0.999),
    )

    score = []
    seen = []

    def callback(i):
        if np.log2(i + 1) % 1.0 == 0.0:
            scr = utils.accuracy(nn(X_test), y_test)
            score.append(scr)
            seen.append((i + 1) * batch_size)

    nn.train(X_train, y_train, n_iter=n_iter, callback=callback)

    print(f"{batch_size} for {n_iter} iterations")

    return score, seen, batch_size


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

    N = X_train.shape[1]

    batch_sizes = [10, 100, 1000]
    # epochs = [1]
    # epochs = [1, 2]

    # build input objects
    params = []
    for bs in batch_sizes:
        # for epoch in epochs:
        params.append(
            {
                "batch_size": bs,
                # "epoch": epoch,
                "n_iter": np.int32((N / bs)) * 20,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }
        )

    # Train in parallel
    with Pool(3) as p:
        results = list(p.map(_train, params))

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    colors = ["lightcoral", "crimson", "darkred"]

    for i, (score, seen, batch_size) in enumerate(results):
        ax.plot(seen, score, label=f"$n_b = {batch_size}$", c=colors[i])

    ax.set_title("Accuracy during training")
    ax.set_xlabel("Images seen")
    ax.set_xscale("log")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(os.path.join(utils.FIGURES_URL, "batches"))
    plt.close(fig)
