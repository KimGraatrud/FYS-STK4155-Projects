import numpy as np


def OLS(X, y):
    # TODO: catch non-invertable
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def ridge(X, y, lam=1e-3):
    return np.linalg.pinv(X.T @ X + lam * np.identity(X.shape[1])) @ X.T @ y
