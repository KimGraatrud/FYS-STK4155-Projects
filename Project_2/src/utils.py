import numpy as np
from sklearn import model_selection

FIGURES_URL = "./figures/"
DATA_URL = "./data/"
SEED = 5318008

rng = np.random.default_rng(seed=SEED)


def runge_1d(x):
    return 1 / (1 + 25 * x**2)


def generate_regression_data(N=1000, noise_std=0.1):
    x = np.linspace(-1, 1, N)
    y_base = runge_1d(x)
    noise = rng.normal(0, noise_std, size=N)
    y = y_base + noise
    return (x, y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


def ReLU(x):
    return np.where(x > 0, x, 0)


def ReLU_der(x):
    return np.where(x > 0, 1, 0)


def mse(predict, target):
    return np.sum((target - predict) ** 2) / len(target)


def mse_der(predict, target):
    return (2 / len(predict)) * (predict - target)




def train_test_split(x, y):
    return model_selection.train_test_split(x, y, train_size=0.8, random_state=SEED)
