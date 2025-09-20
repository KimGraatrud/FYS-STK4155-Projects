import numpy as np


def OLS(X, y):
    # TODO: catch non-invertable
    return np.linalg.inv(X.T @ X) @ X.T @ y
