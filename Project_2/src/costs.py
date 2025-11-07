import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    exp = np.exp(-x)  # for performance, only calc. once

    return exp / (1 + exp) ** 2


def ReLU(x):
    return np.where(x > 0, x, 0)


def ReLU_der(x):
    return np.where(x > 0, 1, 0)


def LeakyReLU(x, alpha=0.05):
    return np.where(x > 0, x, alpha * x)


def LeakyReLU_der(x, alpha=0.05):
    return np.where(x > 0, 1, alpha)


def one(x):
    return x


def one_der(x):
    return np.ones_like(x)


def L1_der(beta, lam=1):
    return lam * np.sign(beta)


def L2_der(beta, lam=1):
    return lam * 2 * beta


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=0)


def mse(predict, target):
    return np.sum((target - predict) ** 2) / np.size(target)


def mse_der(predict, target):
    return (2 / np.size(predict)) * (predict - target)
