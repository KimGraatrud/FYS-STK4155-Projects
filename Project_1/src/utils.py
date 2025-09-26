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

def scale_poly_features(X):
    """
    Scales the columns of the Feature Matrix X using the standard X - mean / std formula.

    Parameter
    ---------
    X : array-like, shape=(n_samples, n_features)

    Returns
    -------
    X_scaled : ndarray
        The scaled feature matrix
    mean : ndarray,
        The means for each column
    nonzero_std : ndarray
        The std of every column with 0 replaced with 1.0
    """

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    nonzero_std = np.where(std == 0, 1., std)
    X_scaled = (X - mean) / nonzero_std
    scaler = (mean, nonzero_std)
    return X_scaled, scaler

def runge(x):
    return 1 / (1 + 25 * x**2)


FIGURES_URL = "./figures/"
RANDOM_SEED = 2025