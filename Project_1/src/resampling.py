import numpy as np
from sklearn.model_selection import train_test_split
from . import utils, regression, ml
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import make_scorer


class resampling_methods:
    """
    Class containing all of the resampling methods used for generating the results of the proejct.
    """
    def __init__(self, x, y, maxdegree, lamb_arr_R, lamb_arr_L):
        self._lamb_R = lamb_arr_R
        self._lamb_L = lamb_arr_L
        self.degrees = np.arange(maxdegree+1)
        self.x = x
        self.y = y
        self.rng = np.random.default_rng(seed=utils.RANDOM_SEED)

    def _calc_MSE_bias_var(self, truth: np.ndarray, prediction: np.ndarray):
        """
        Calculates the MSE bias^2 and variance using self.y as the truth and prediction as the prediction over every bootstrap.
        """

        MSE = np.mean( np.mean((truth - prediction)**2, axis=1, keepdims=True) )
        bias2 = np.mean( (truth - np.mean(prediction, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(prediction, axis=1, keepdims=True) )

        return MSE, bias2, variance

    def _indices_k_split(self, n: int, k: int, rng, verbose=False):
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

    def Bootstrap_one_deg(self, x, y,
                        deg,
                        nbootstraps,
                        replacement=True, 
                        verbose: bool = False):
        """
        Run boostrap for one degree. Used to study the effect of number of datapoints on the bias-variance tradeoff
        """

        X = utils.poly_features(x, deg, intercept=True)

        y = y.reshape(-1,1)
        Xtrain, Xtest, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=utils.RANDOM_SEED
        )
        n_train = len(y_train)
        n_test = len(y_test)

        OLSprediction_train = np.zeros((n_train, nbootstraps), dtype=float)
        OLSprediction_test = np.zeros((n_test, nbootstraps), dtype=float)

        for j in range(nbootstraps):

            bootstrap_idx = self.rng.choice(n_train, size=n_train, replace=replacement)
            Xtrain_S = Xtrain[bootstrap_idx]
            ytrain_S = y_train[bootstrap_idx]
            
            OLSbeta = regression.OLS(Xtrain_S, ytrain_S)
            OLSprediction_train[:,j] = (Xtrain @ OLSbeta).ravel()
            OLSprediction_test[:,j] = (Xtest @ OLSbeta).ravel()

        trMSE, trBias, trVar = self._calc_MSE_bias_var(y_train, OLSprediction_train)
        teMSE, teBias, teVar = self._calc_MSE_bias_var(y_test, OLSprediction_test)

        if verbose:
            print(trMSE, trBias, trVar)
            print(teMSE, teBias, teVar)

        return trMSE, trBias, trVar, teMSE, teBias, teVar

    def BootstrapOLS(self, nbootstraps: int, replacement=True,verbose: bool = False):

        degrees = self.degrees
        nDeg = len(degrees)

        y = self.y.reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, y, train_size=0.8, random_state=utils.RANDOM_SEED
        )
        n_train = len(y_train)
        n_test = len(y_test)


        OLSMSE_train = np.empty(nDeg, dtype=float)
        OLSbias2_train = np.empty(nDeg, dtype=float)
        OLSvariance_train = np.empty(nDeg, dtype=float)

        OLSMSE_test = np.empty(nDeg, dtype=float)
        OLSbias2_test = np.empty(nDeg, dtype=float)
        OLSvariance_test = np.empty(nDeg, dtype=float)


        for i, deg in enumerate(degrees):

            OLSprediction_train = np.zeros((n_train, nbootstraps), dtype=float)
            OLSprediction_test = np.zeros((n_test, nbootstraps), dtype=float)

            Xtrain = utils.poly_features(x_train, deg, intercept=True)
            Xtest = utils.poly_features(x_test, deg, intercept=True)

            for j in range(nbootstraps):

                bootstrap_idx = self.rng.choice(n_train, size=n_train, replace=replacement)
                Xtrain_S = Xtrain[bootstrap_idx]
                ytrain_S = y_train[bootstrap_idx]
                
                OLSbeta = regression.OLS(Xtrain_S, ytrain_S)
                OLSprediction_train[:,j] = (Xtrain @ OLSbeta).ravel()
                OLSprediction_test[:,j] = (Xtest @ OLSbeta).ravel()

                if verbose:
                    print('Bootstrap j:', j, 'ok.')

            OLSMSE_train[i], OLSbias2_train[i], OLSvariance_train[i] = self._calc_MSE_bias_var(y_train, OLSprediction_train)
            OLSMSE_test[i], OLSbias2_test[i], OLSvariance_test[i] = self._calc_MSE_bias_var(y_test, OLSprediction_test)
            
            if verbose:
                print('Degree i:', i, 'ok.')
                print(f"Degree {deg}:")
                print(OLSMSE_train[i],OLSMSE_test[i])
                print(OLSbias2_train[i],OLSbias2_test[i])
                print(OLSvariance_train[i],OLSvariance_test[i])

        result = {
            'Train': (OLSMSE_train, OLSbias2_train, OLSvariance_train),
            'Test': (OLSMSE_test, OLSbias2_test, OLSvariance_test)
        }

        return self.degrees, result

    def k_fold_CV_OLS(self, k: int, verbose: bool = False):

        nsamples = len(self.y)
        indices, folds = self._indices_k_split(nsamples, k, self.rng)
        degrees = self.degrees
        ndeg = len(degrees)

        OLSscores_train = np.empty((ndeg, k), dtype=float)
        OLSscores_test = np.empty((ndeg, k), dtype=float)

        for i, deg in enumerate(degrees):

            X = utils.poly_features(self.x, deg, intercept=True)

            for j, test_idx in enumerate(folds):

                # Grab every index exept for the current fold
                mask = np.isin(indices, test_idx, assume_unique=True)
                train_idx = indices[~mask]

                Xtrain = X[train_idx]
                Xtest = X[test_idx]
                y_train = self.y[train_idx]
                y_test = self.y[test_idx]

                OLS_beta = regression.OLS(Xtrain, y_train)
                prediction_train = (Xtrain @ OLS_beta).ravel()
                prediction_test = (Xtest @ OLS_beta).ravel()

                OLSscores_train[i, j] = np.sum((prediction_train - y_train)**2)/np.size(prediction_train)
                OLSscores_test[i, j] = np.sum((prediction_test - y_test)**2)/np.size(prediction_test)

                if verbose:
                    print(OLSscores_train[i,j])
                    print(OLSscores_test[i,j])

        OLS_MSE_train = np.mean(OLSscores_train, axis=1)
        OLS_MSE_test = np.mean(OLSscores_test, axis=1)

        return OLS_MSE_train, OLS_MSE_test


    def k_fold_CV_ALL(self, k: int,
                verbose: bool = False):
        """
        Runs k-fold-CV on OLS, Ridge, and Lasso, reports the MSE of all methods for each degree.
        """

        degrees = self.degrees
        lamb_R = self._lamb_R
        lamb_L = self._lamb_L
        indices, folds = self._indices_k_split(len(self.y), k, self.rng)
        ndeg = len(degrees)

        OLS_MSE = np.empty_like(degrees, dtype=float)
        Ridge_MSE = np.empty((ndeg, len(lamb_R)), dtype=float)
        Lasso_MSE = np.empty((ndeg, len(lamb_L)), dtype=float)
        Ridge_param = np.empty_like(degrees, dtype=float)
        Lasso_param = np.empty_like(degrees, dtype=float)
        Ridge_param_idx = np.empty_like(degrees, dtype=int)
        Lasso_param_idx = np.empty_like(degrees, dtype=int)

        for i, deg in enumerate(degrees):

            OLSfoldMSE = np.empty(k, dtype=float)
            RidgefoldMSE = np.empty((len(lamb_R), k), dtype=float)
            LassofoldMSE = np.empty((len(lamb_L), k), dtype=float)

            for j, test_idx in enumerate(folds):

                # Grab every index exept for the current fold
                mask = np.isin(indices, test_idx, assume_unique=True)
                train_idx = indices[~mask]

                Xtrain = utils.poly_features(self.x[train_idx], deg, intercept=True)
                Xtest = utils.poly_features(self.x[test_idx], deg, intercept=True)
                y_train = self.y[train_idx]
                y_test = self.y[test_idx]

                OLS_beta = regression.OLS(Xtrain, y_train)
                OLS_prediction = Xtest @ OLS_beta
                OLSfoldMSE[j] = utils.MSE(y_test, OLS_prediction)

                for ri, lamb in enumerate(lamb_R):

                    Ridge_beta = regression.ridge(Xtrain, y_train, lam=lamb, has_intercept=True)
                    Ridge_prediction = Xtest @ Ridge_beta
                    RidgefoldMSE[ri, j] = utils.MSE(y_test, Ridge_prediction)

                for li, lamb in enumerate(lamb_L):

                    _iter = 1e4
                    model = ml.GD(Xtrain, y_train, n_iterations=_iter, lamb=lamb, eta=1e-1, has_interecpt=True)
                    Lasso_beta = model.Lasso()
                    Lasso_prediction = Xtest @ Lasso_beta
                    LassofoldMSE[li, j] = utils.MSE(y_test, Lasso_prediction)

                if verbose:
                    print(i, j)

            OLS_MSE[i] = np.mean(OLSfoldMSE)
            R_mean = np.mean(RidgefoldMSE, axis=1)
            L_mean = np.mean(LassofoldMSE, axis=1)
            Ridge_MSE[i, :] = R_mean
            Lasso_MSE[i, :] = L_mean

            best_Ridge_param = np.argmin(R_mean)
            Ridge_param[i] = lamb_R[best_Ridge_param]
            best_Lasso_param = np.argmin(L_mean)
            Lasso_param[i] = lamb_L[best_Lasso_param]

            Ridge_param_idx[i] = best_Ridge_param
            Lasso_param_idx[i] = best_Lasso_param

        Ridge_MSE = Ridge_MSE[np.arange(len(degrees)), Ridge_param_idx]
        Lasso_MSE = Lasso_MSE[np.arange(len(degrees)), Lasso_param_idx]

        return OLS_MSE, Ridge_MSE, Lasso_MSE, Ridge_param, Lasso_param
