import os
import numpy as np
import matplotlib.pyplot as plt
from src import utils
from src.FFNN import FFNN


def main():
    x, y = utils.generate_regression_data()

    x_train, x_test, y_train, y_test = utils.train_test_split(x, y)

    x_mat = np.expand_dims(x_train, axis=1)
    y_mat = np.expand_dims(y_train, axis=1)
    print("x_mat", x_mat)

    nn = FFNN(
        network_input_size=1,
        layer_output_sizes=[5, 5, 1],
        activation_funcs=[utils.ReLU, utils.ReLU, utils.ReLU_der],
        activation_ders=[utils.ReLU_der, utils.ReLU_der, utils.ReLU_der],
        cost_fun=utils.mse,
        cost_der=utils.mse_der,
    )

    nn.train(x_mat, y_mat)

    y_pred = [nn(x) for x in x_test]

    fig, ax = plt.subplots()

    fig.savefig()


if __name__ == "__main__":
    main()
