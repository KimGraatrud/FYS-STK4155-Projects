import numpy as np
from sklearn.model_selection import train_test_split
from . import utils, regression
from sklearn.utils import resample



class resampling_methods:
    """
    Class containing all of the resampling methods used for generating the results of the proejct.
    """
    def __init__(self, x, y, maxdegree, lamb_arr_R, lamb_arr_L):
        self._lamb_R = lamb_arr_R
        self._lamb_L = lamb_arr_L
        self.degrees = np.arange(maxdegree+1)
        self.x = x
        self.y = y.reshape(-1, 1)
        self.rng = np.random.default_rng(seed=utils.RANDOM_SEED)

    def _calc_MSE_bias_var(truth: np.ndarray, prediction: np.ndarray):
        """
        Calculates the MSE bias^2 and variance using self.y as the truth and prediction as the prediction over every bootstrap.
        """

        MSE = np.mean( np.mean((self.y - prediction)**2, axis=1, keepdims=True) )
        bias2 = np.mean( (self.y - np.mean(prediction, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(prediction, axis=1, keepdims=True) )

        return MSE, bias2, variance

    def BootstrapALL(self,
                n_bootstraps: int,
                verbose: bool = False) -> np.ndarray:
        # """
        # Preforms bootstrap over model complexity and calculates the MSE, bias, variance for each degree.
        # """

        degrees = self.degrees
        n_test = len(self.y)
        nR, nL, nDeg = len(self._lamb_R), len(self._lamb_L), len(degrees)
        num_models = 3 # OLS Ridge and Lasso

        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, train_size=0.8, random_state=utils.RANDOM_SEED
        )

        y_col_test = y_test.reshape(-1, 1)
        
        OLSMSE = np.empty(nDeg, dtype=float)
        OLSbias2 = np.empty(nDeg, dtype=float)
        OLSvariance = np.empty(nDeg, dtype=float)

        # Over every degree and lambda value
        RidgeMSE = np.empty((nDeg, nR), dtype=float)
        Ridgebias2 = np.empty((nDeg, nR), dtype=float)
        Ridgevariance = np.empty((nDeg, nR), dtype=float)

        LassoMSE = np.empty((nDeg, nL), dtype=float)
        Lassobias2 = np.empty((nDeg, nL), dtype=float)
        Lassovariance = np.empty((nDeg, nL), dtype=float)

        for i, deg in enumerate(degrees):

            OLSprediction = np.zeros((n_test, n_bootstraps), dtype=float)
            Ridgeprediction = np.zeros((n_test, n_bootstraps, nR), dtype=float)
            Lassoprediction = np.zeros((n_test, n_bootstraps, nL), dtype=float)

            Xtest = utils.poly_features(x_test, deg, intercept=True)

            for j in range(n_bootstraps):
                _x, _y = resample(x_train, y_train)
                Xtrain = utils.poly_features(_x, deg, intercept=True)

                OLSbeta = regression.OLS(Xtrain, _y)
                OLSprediction[:,j] = (Xtest @ OLSbeta).ravel()

                for ri, lamb in enumerate(self._lamb_R):

                    Ridge_beta = regression.ridge(Xtrain, y_train, lam=lamb)
                    Ridgeprediction[:, j, ri] = Xtest @ Ridge_beta

                for li, lamb in enumerate(self._lamb_L):

                    _iter = 1000
                    model = ml.GD(n_iterations=_iter, lamb=lamb)
                    Lasso_beta = model.Lasso(Xtrain, y_train)
                    Lasso_prediction[:, j, li] = Xtest @ Lasso_beta

            OLSMSE[i], OLSbias2[i], OLSvariance[i] = _calc_MSE_bias_var(self.y, OLSprediction)

            for ri in range(nR):
                RidgeMSE[i, ri], Ridgebias2[i, ri], Ridgevariance[i, ri] = _calc_MSE_bias_var(
                    self.y, Ridgeprediction[:, :, ri]
                )

            for li in range(nL):
                LassoMSE[i, li], Lassobias2[i, li], Lassovariance[i, li] = _calc_MSE_bias_var(
                    self.y, Lasso_prediction[:, :, li]
                )

        if verbose:
            print(f"Degree {deg}:")
            print(f"  OLS   -> MSE={OLS_MSE[i]:.6e}  Bias^2={OLS_bias2[i]:.6e}  Var={OLS_var[i]:.6e}")
            for ri, lam in enumerate(lambdas_R):
                print(f"  Ridge λ={lam:.3g} -> MSE={Ridge_MSE[i, ri]:.6e}  Bias^2={Ridge_bias2[i, ri]:.6e}  Var={Ridge_var[i, ri]:.6e}")
            for li, lam in enumerate(lambdas_L):
                print(f"  Lasso λ={lam:.3g} -> MSE={Lasso_MSE[i, li]:.6e}  Bias^2={Lasso_bias2[i, li]:.6e}  Var={Lasso_var[i, li]:.6e}")

        # Return all MSE, Bias^2 and variance values for all methods
        OLS_result = {
            'MSE': OLSMSE,
            'Bias2': OLSbias2, 
            'Variance': OLSvariance
        }
        Ridge_stats = {
            'MSE': RidgeMSE,
            'Bias2': Ridgebias2, 
            'Variance': Ridgevariance
        }
        Lasso_stats = {
            'MSE': LassoMSE,
            'Bias2': Lassobias2, 
            'Variance': Lassovariance
        }

        return self.degrees, OLS_stats, Ridge_stats, Lasso_stats

    
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

    def k_fold_CV(self, k: int,
                verbose: bool = False):
        """
        Runs k-fold-CV on OLS, Ridge, and Lasso, reports the MSE of all methods for each degree.
        """

        degrees = self.degrees
        lamb_R = self._lamb_R
        lamb_L = self._lamb_L
        indices, folds = _indencies_k_split(len(self.y), k, self.rng)
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

                Xtrain = utils.poly_features(self.x[train_idx], deg, intercept=True)
                Xtest = utils.poly_features(self.x[test_idx], deg, intercept=True)
                y_train = self.y[train_idx]
                y_test = self.y[test_idx]

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

        
