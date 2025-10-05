import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from .. import utils, regression
from .A import create_data

def _indices_k_split(n: int, k: int, rng, verbose=False):
    """
    Split indices randomly and assign one block to train and the rest to train.
    """

    indices = np.arange(n)
    rng.shuffle(indices)
    split = np.array_split(indices, k)

    if verbose:
        print(indices)
        print(split)

    return indices, split

def k_fold_CV(x: int, y: int, k: int,
              degrees: np.ndarray,
              lamb_R: np.ndarray,
              lamb_L: np.ndarray,
              verbose: bool = False):
    """
    Runs k-fold-CV on OLS, Ridge, and Lasso, reports the MSE of all methods for each degree.
    """

    rng = np.random.default_rng(seed=utils.RANDOM_SEED)
    indices, folds = _indencies_k_split(len(y), k, rng)
    ndeg = len(degrees)

    OLS_MSE = np.emptylike(degrees, dtype=float)
    Ridge_MSE = np.empty((ndeg, len(lamb_R)), dtype=float)
    Lasso_MSE = np.empty((ndeg, len(lamb_L)), dtype=float)
    Ridge_param = np.emptylike(degrees, dtype=float)
    Lasso_param = np.emptylike(degrees, dtype=float)

    for i, deg in enumerate(degrees):

        OLSfoldMSE = np.empty(k, dtype=float)
        RidgefoldMSE = np.empty((len(lamb_R), k), dtype=float)
        LassofoldMSE = np.empty((len(lamb_L), k), dtype=float)

        for j, test_idx in enumerate(folds):

            # Grab every index exept for the current fold
            mask = np.isin(indices, test_idx, assume_unique=True)
            train_idx = indices[~mask]

            Xtrain = utils.poly_features(x[train_idx], deg, intercept=True)
            Xtest = utils.poly_features(x[test_idx], deg, intercept=True)
            y_train = y[train_idx]
            y_test = y[test_idx]

            OLS_beta = regression.OLS(Xtrain, y_train)
            OLS_prediction = Xtest @ OLS_beta
            OLSfoldMSE[j] = utils.MSE(y_test, OLS_prediction)

            for ri, lamb in enumerate(lamb_R):

                Ridge_beta = regression.ridge(Xtrain, y_train, lam=lamb)
                Ridge_prediction = Xtest @ Ridge_beta
                RidgefoldMSE[ri, j] = utils.MSE(y_test, Ridge_prediction)

            for li, lamb in enumerate(lamb_L):

                _iter = 1000
                model = ml.GD(n_iterations=_iter, lamb=lamb)
                Lasso_beta = model.Lasso(Xtrain, y_train)
                Lasso_prediction = Xtest @ Lasso_beta
                LassofoldMSE[li, j] = utils.MSE(y_test, Lasso_prediction)

        OLS_MSE[i] = np.mean(OLSfoldMSE)
        Ridge_MSE[i, :] = np.mean(RidgefoldMSE, axis=1)
        Lasso_MSE[i, :] = np.mean(LassofoldMSE, axis=1)

        best_Ridge_param = np.argmin(RidgefoldMSE, axis=0)
        Ridge_param[i] = lamb_R[best_Ridge_param]
        best_Lasso_param = np.argmin(LassofoldMSE, axis=0)
        Lasso_param[i] = lamb_R[best_Lasso_param]

    if verbose:
        print(OLS_MSE)
        print(Ridge_MSE)
        print(Lasso_MSE)
        print(Ridge_param)
        print(Lasso_param)

    return OLS_MSE, Ridge_MSE, Lasso_MSE, Ridge_param, Lasso_param


    

def main():

    n = 100
    k = 10



if __name__ == '__main__':
    main()

    
