import os
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from src.ClassifierNN import ClassifierNN
from src import utils, costs


def _classify(params):
    data, method = params
    X, X_test, y, y_test = utils.prep_classification_data(data)

    batch_size = 16

    nn = ClassifierNN(
        network_input_size=X.shape[0],
        classes=y.shape[0],
        layer_output_sizes=[100],
        activation_funcs=[costs.LeakyReLU],
        activation_ders=[costs.LeakyReLU_der],
        batch_size=batch_size,
        regularization_der=None,
        **method,
    )

    score = []
    seen = []

    n_iter = 1e4

    recording_points = np.int32(np.logspace(0, np.log10(n_iter), 40))

    def callback(i):
        # logging
        if i % 1000 == 0:
            print(f"{i} {utils.accuracy(nn(X_test), y_test):.3f}")

        # recording
        if i in recording_points:
            scr = utils.accuracy(nn(X_test), y_test)
            score.append(scr)
            seen.append((i + 1) * batch_size)

    nn.train(X, y, n_iter=n_iter, callback=callback)

    return seen, score


def main():
    # Fetch the MNIST dataset
    mnist = utils.load_openml_dataset()

    with Pool(3) as p:
        params = [
            (mnist, {"eta": 0.5}),
            (
                mnist,
                {
                    "eta": 1e-2,
                    "descent_method": "rmsprop",
                    "decay_rate": 0.99,
                },
            ),
            (
                mnist,
                {
                    "eta": 1e-2,
                    "descent_method": "adam",
                    "decay_rate": (0.9, 0.999),
                },
            ),
        ]
        results = p.map(_classify, params)

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    ax.plot(results[0][0], results[0][1], label="GD")
    ax.plot(results[1][0], results[1][1], label="RMSProp")
    ax.plot(results[2][0], results[2][1], label="ADAM")

    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.legend()

    ax.set_xlabel("Images seen")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Behavior of Each Gradient Descent Method")

    fig.savefig(os.path.join(utils.FIGURES_URL, "descent"))
    plt.close(fig)

    # Confusion matrix
