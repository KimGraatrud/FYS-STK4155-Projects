import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from .. import utils, regression, resampling
from .A import create_data


def plot_bootstrapdegree(degrees:   np.ndarray,
                         MSE:       np.ndarray,
                         bias:      np.ndarray,
                         variance:  np.ndarray
                         ) -> None:

    """
    Simple helper func for plotting bootstrap MSE over degrees. 
    """

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

def plot_bias_var_tradeoff(degrees:   np.ndarray,
                         MSE:       np.ndarray,
                         bias:      np.ndarray,
                         variance:  np.ndarray
                         ) -> None:

    """
    Plots the bias-variance tradeoff for OLS 
    """

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("$\hat{MSE}$ | Bias | Var")

    cmap = plt.colormaps["Reds"]
    norm = mpl.colors.Normalize(vmin=0.5, vmax=1.2)

    ax.set_title("Bias-Variance Tradeoff")
    ax.set_yscale('log')

    ax.plot(degrees, MSE, label='$\hat{MSE}$')
    ax.plot(degrees, bias, label='Bias')
    ax.plot(degrees, variance, label='Var')

    ax.legend()
    fig.set_figheight(0.9 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "BiasVarianceTradeoff"))
    plt.close()

def bias_var_over_points():

    min_points = 2
    max_points = 4
    num_points = 100
    datapoints = np.logspace(min_points, max_points, num_points)
    degree       = 5
    nbootstrap = int(1e3)

    trMSE, trBias, trVar = np.empty(num_points), np.empty(num_points), np.empty(num_points)
    teMSE, teBias, teVar = np.empty(num_points), np.empty(num_points), np.empty(num_points)

    for i, points in enumerate(datapoints):
            
        x, y = create_data(points)
        strap = resampling.resampling_methods(x, y, 0, 0, 0)
        nbootstraps = int(1e3)
        trMSE[i], trBias[i], trVar[i], teMSE[i], teBias[i], teVar[i] = strap.Bootstrap_one_deg(
            x, y, degree, nbootstrap, replacement=True, verbose=True
        )

    # Using our standard figuresize
    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    ax.set_xlabel("Number of datapoints")
    ax.set_ylabel("$\hat{MSE}$ | Bias | Var")

    cmap = plt.colormaps["Reds"]
    norm = mpl.colors.Normalize(vmin=0.5, vmax=1.2)

    ax.set_title("Bias-Variance")
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.plot(datapoints, teMSE, label='MSE')
    ax.plot(datapoints, teBias, label='Bias', linestyle='dashed')
    ax.plot(datapoints, teVar, label='Var')

    ax.legend()
    fig.set_figheight(0.9 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "BiasVarianceTradeoffNPoints"))
    plt.close()

def bias_var_over_deg():

    n_datapoints = int(1e2)
    maxdeg       = 10
    x, y = create_data(n_datapoints)

    strap = resampling.resampling_methods(x, y, maxdeg, 0, 0)

    nbootstraps = int(1e3)
    degrees, Boot_result = strap.BootstrapOLS(
        nbootstraps, verbose=False, replacement=True
    )

    trMSE, trBias, trVar = Boot_result['Train']
    teMSE, teBias, teVar = Boot_result['Test']


    plot_bias_var_tradeoff(degrees, teMSE, teBias, teVar)



def main():
    bias_var_over_points()
    bias_var_over_deg()

if __name__ == "__main__":
    main()
