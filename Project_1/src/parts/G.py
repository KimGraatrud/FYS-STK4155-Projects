import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from .. import utils, regression
from .A import create_data


def plot_bootstrapdegree(degrees:   np.ndarray,
                         MSE:       np.ndarray,
                         bias:      np.ndarray,
                         variance:  np.ndarray
                         ) -> None:

    # Using our standard figuresize
    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    ax.set_xlabel("Degree")
    ax.set_ylabel("MSE | Bias | $\sigma^2$")

    cmap = plt.colormaps["Reds"]
    norm = mpl.colors.Normalize(vmin=0.5, vmax=1.2)

    ax.set_title("Bias-Variance")

    ax.plot(degrees, MSE, label='MSE')
    ax.plot(degrees, bias, label='Bias', linestyle='dashed')
    ax.plot(degrees, variance, label='Variance')

    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.set_figheight(0.9 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "BiasVarianceTradeoff"))
    plt.close()



def BootstrapOLS(n_bootstraps:  int,
                 maxdegree:     int,
                 x:             np.ndarray,
                 y:             np.ndarray,
                 show_output:   bool = False,
                 ) -> np.ndarray:

    degrees = np.arange(maxdegree + 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=utils.RANDOM_SEED
    )
    
    MSE = np.empty_like(degrees, dtype=float)
    bias = np.empty_like(degrees, dtype=float)
    variance = np.empty_like(degrees, dtype=float)

    for i, deg in enumerate(degrees):

        prediction = np.zeros((len(y_test), n_bootstraps), dtype=float)

        Xtest = utils.poly_features(x_test, deg, intercept=True)

        for j in range(n_bootstraps):
            _x, _y = resample(x_train, y_train)
            Xtrain = utils.poly_features(_x, deg, intercept=True)
            beta = regression.OLS(Xtrain, _y)
            prediction[:,j] = (Xtest @ beta).ravel()

        
        y_col = y_test.reshape(-1, 1)
        MSE[i] = np.mean( np.mean((y_col - prediction)**2, axis=1, keepdims=True) )
        bias[i]  = np.mean( (y_col - np.mean(prediction, axis=1, keepdims=True))**2 )
        variance[i] = np.mean( np.var(prediction, axis=1, keepdims=True) )

        if show_output:
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
    


def main():
    
    n_datapoints = int(1e3)
    maxdeg       = 18
    x, y = create_data(n_datapoints)

    n_bootstrap = int(1e2)
    degrees, MSE, bias, variance = BootstrapOLS(
        n_bootstrap, maxdeg, x, y, show_output=True
    )

    plot_bootstrapdegree(degrees, MSE, bias, variance)
    


if __name__ == "__main__":
    main()
