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


def main():
    
    n_datapoints = int(1e3)
    maxdeg       = 18
    x, y = create_data(n_datapoints)

    strap = resampling.resampling()

    n_bootstrap = int(1e2)
    degrees, MSE, bias, variance = strap.BootstrapOLS(
        n_bootstrap, maxdeg, x, y, verbose=True
    )

    plot_bootstrapdegree(degrees, MSE, bias, variance)

if __name__ == "__main__":
    main()
