import numpy as np


def OLS(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def ridge(X, y, lam=1e-3, has_intercept=True):
    n,p = X.shape
    S = np.identity(p)
    if has_intercept:
        S[0,0] = 0.0
    return np.linalg.pinv(X.T @ X + lam * S) @ X.T @ y
