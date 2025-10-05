import numpy as np
from sklearn.model_selection import train_test_split
from . import utils, regression
from sklearn.utils import resample


class resampling:
    """
    Class containing all of the resampling methods used for generating the results of the proejct.
    """
    def __init__(self):
        pass

    def BootstrapOLS(self,
                n_bootstraps: int,
                maxdegree: int,
                x: np.ndarray,
                y: np.ndarray,
                verbose: bool = False
                ) -> np.ndarray:
        """
        Preforms bootstrap over model complexity and calculates the MSE, bias, variance for each degree.
        """

        degrees = np.arange(maxdegree + 1)
        num_models = 3 # OLS Ridge and Lasso

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=0.8, random_state=utils.RANDOM_SEED
        )
        
        MSE = np.empty_like((degrees, num_models), dtype=float)
        bias = np.empty_like((degrees, num_models), dtype=float)
        variance = np.empty_like((degrees, num_models), dtype=float)

        for i, deg in enumerate(degrees):

            prediction = np.zeros((len(y_test), n_bootstraps), dtype=float)

            Xtest = utils.poly_features(x_test, deg, intercept=True)

            for j in range(n_bootstraps):
                _x, _y = resample(x_train, y_train)
                Xtrain = utils.poly_features(_x, deg, intercept=True)
                OLSbeta = regression.OLS(Xtrain, _y)
                prediction[:,j] = (Xtest @ beta).ravel()

            
            y_col = y_test.reshape(-1, 1)
            MSE[i] = np.mean( np.mean((y_col - prediction)**2, axis=1, keepdims=True) )
            bias[i]  = np.mean( (y_col - np.mean(prediction, axis=1, keepdims=True))**2 )
            variance[i] = np.mean( np.var(prediction, axis=1, keepdims=True) )

            if verbose:
                print('Polynomial degree:', deg)
                print('Error:', MSE[i])
                print('Bias^2:', bias[i])
                print('Var:', variance[i])
                print('{} >= {} + {} = {}'.format(MSE[i],
                        bias[i], 
                        variance[i], 
                        bias[i]+variance[i])
                    )

        return degrees, MSE, bias, variance
    
