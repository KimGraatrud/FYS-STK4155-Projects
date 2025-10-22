import os
import numpy as np
import matplotlib.pyplot as plt
from src import utils
from src.FFNN import FFNN


def main():
    x, y = utils.generate_regression_data(N=1000)

    x_train, x_test, y_train, y_test = utils.train_test_split(x, y)

    x_mat = np.expand_dims(x_train, axis=0)
    y_mat = np.expand_dims(y_train, axis=0)

    nn = FFNN(
        network_input_size=x_mat.shape[0],
        layer_output_sizes=[3, 3, y_mat.shape[0]],
        activation_funcs=[utils.sigmoid, utils.ReLU, utils.sigmoid],
        activation_ders=[utils.sigmoid_der, utils.ReLU_der, utils.sigmoid_der],
        cost_fun=utils.mse,
        cost_der=utils.mse_der,
        eta=1e-2,
    )

    nn.train(x_mat, y_mat)

    print("train!\n\n")

    x_test_mat = np.expand_dims(x_test, axis=0)
    y_pred = nn(x_test_mat)

    fig, ax = plt.subplots()
    ax.scatter(x_test, y_test, s=0.5)
    ax.scatter(x_test, y_pred.flatten())
    fig.savefig(os.path.join(utils.FIGURES_URL, "test"))


if __name__ == "__main__":
    main()
