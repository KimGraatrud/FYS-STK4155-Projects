import numpy as np


def MSE(truth, prediction):
    return np.sum((truth - prediction) ** 2) / truth.shape[0]


def Rsqd(truth, prediction):
    sum1 = np.sum((truth - prediction) ** 2)
    sum2 = np.sum((truth - np.mean(truth)) ** 2)
    return 1 - (sum1 / sum2)


def poly_features(x, d, intercept=False):
    """
    x: input
    d: degree
    """
    lowest = 0 if intercept else 1
    orders = np.arange(lowest, d + 1)
    exp, base = np.meshgrid(orders, x)
    return base**exp
