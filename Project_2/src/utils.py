import numpy as np
from sklearn import model_selection

FIGURES_URL = "./figures/"
DATA_URL = "./data/"
SEED = 5318008

rng = np.random.default_rng(seed=SEED)


def runge_1d(x):
    return 1 / (1 + 25 * x**2)


def runge_2d(x, y):
    return 1 / ((10 * x - 5) ** 2 + (10 * y - 5) ** 2 + 1)


def train_test_split(x, y, **kwargs):
    return model_selection.train_test_split(
        x, y, train_size=0.8, random_state=SEED, **kwargs
    )


def generate_regression_data(N=1000, noise_std=0.1, dim=1):
    """
    dim: either 1 or 2, corresponding to the 1D or 2D Runge function
    """
    x = np.linspace(-1, 1, N)

    if dim == 1:
        y_base = runge_1d(x)
        x = np.expand_dims(x, axis=0)
    elif dim == 2:
        x1, x2 = np.meshgrid(x, x)
        x = np.array([np.ravel(x1), np.ravel(x2)])
        y_base = runge_2d(x[0], x[1])
    else:
        raise ValueError(f"Invalid input dimension: {dim}; must be 1 or 2")

    # add noise
    noise = rng.normal(0, noise_std, size=len(y_base))
    y = y_base + noise

    y = np.expand_dims(y, axis=0)

    return (x, y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    # for performance, only calc. once
    exp = np.exp(-x)

    return exp / (1 + exp) ** 2


def ReLU(x):
    return np.where(x > 0, x, 0)


def ReLU_der(x):
    return np.where(x > 0, 1, 0)


def mse(predict, target):
    return np.sum((target - predict) ** 2) / len(target)


def mse_der(predict, target):
    return (2 / len(predict)) * (predict - target)
