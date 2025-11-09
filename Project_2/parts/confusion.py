import os
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.FFNN import FFNN
from src.ClassifierNN import ClassifierNN
from src import utils, costs


def _classify(data):
    X, X_test, y, y_test = utils.prep_classification_data(data)

    nn = ClassifierNN(
        network_input_size=X.shape[0],
        classes=y.shape[0],
        layer_output_sizes=[100],
        activation_funcs=[costs.LeakyReLU],
        activation_ders=[costs.LeakyReLU_der],
        eta=1e-3,
        batch_size=16,
        regularization_der=None,
        descent_method="adam",
        decay_rate=(0.9, 0.999),
    )

    nn.train(X, y, n_iter=1e5)

    return nn(X_test), y_test


def main():
    # Fetch the MNIST dataset
    mnist = utils.load_openml_dataset()
    fashion = utils.load_openml_dataset(dataset="fashion-mnist")

    with Pool(2) as p:
        results = p.map(_classify, [mnist, fashion])

    fig, axs = plt.subplots(nrows=2, figsize=(utils.APS_COL_W, 2 * utils.APS_COL_W))

    matrices = []

    for result in results:
        pred, truth = result

        print(f"Accuracy: {utils.accuracy(pred, truth)}")

        matrix = np.zeros((10, 10))

        for p, t in zip(pred.T, truth.T):
            matrix[np.argmax(t), np.argmax(p)] += 1

        matrices.append(matrix)

    matrices = np.array(matrices)
    matrices[matrices == 0] = np.nan

    norm = mpl.colors.LogNorm(vmin=np.nanmin(matrices), vmax=np.nanmax(matrices))

    cmap = plt.colormaps["viridis"]
    cmap.set_bad("k")

    for i, matrix in enumerate(matrices):
        im = axs[i].imshow(matrix, norm=norm, cmap=cmap)

    names = [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    ticks = np.arange(10)
    axs[0].set_xticks(ticks)
    axs[0].set_yticks(ticks)
    axs[0].set_xlabel("Prediction")
    axs[0].set_ylabel("Actual")
    axs[1].set_xlabel("Prediction")
    axs[1].set_ylabel("Actual")

    axs[0].set_title("MNIST")
    axs[1].set_title("Fashion")

    axs[1].set_xticks(
        ticks,
        labels=names,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )

    axs[1].set_yticks(ticks, labels=names)

    fig.colorbar(im, ax=axs[1], location="bottom", label=r"\# of occurances")

    fig.savefig(os.path.join(utils.FIGURES_URL, "confusion"))
    plt.close(fig)

    # Confusion matrix
