import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from .utils import poly_features, MSE, runge, scale_poly_features
from .utils import RANDOM_SEED, FIGURES_URL

def OLS(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def RidgeRegression(X, y, lmbda=1e-3):
    n,p = X.shape; I = np.eye(p)
    return np.linalg.pinv(X.T @ X + n*lmbda*I) @ X.T @ y


def Heatmap_vary_ridgeparam():

    np.random.seed(RANDOM_SEED)

    x = np.linspace(-1,1, 100)
    y_true = runge(x)
    y = y_true + np.random.normal(0, 0.1, x.shape[0])
    y_centered = y - np.mean(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.2, random_state=RANDOM_SEED, shuffle=True
    )
    
    n = 50
    lambda_vals = np.logspace(0, -5, n)

    maxdeg = 10
    degrees = np.arange(1, maxdeg+1, 1)

    data = np.empty(shape=(len(lambda_vals), len(degrees)))


    for i, deg in enumerate(degrees):

        Xtr = poly_features(x, deg)
        Xtr_s, scaler = scale_poly_features(Xtr)

        for j, lmbda in enumerate(lambda_vals):
 
            beta = RidgeRegression(Xtr_s, y_centered, lmbda=lmbda)
            prediction = Xtr_s @ beta
            data[j,i] = MSE(y_true, prediction)

        
    # ------------ Plot it all -------------
    fig, ax = plt.subplots(figsize=(6, 4))

    # MSE
    cmap_mse = plt.colormaps["viridis"]
    cmap_mse.set_bad(color="dimgray")

    ds, ls = np.meshgrid(lambda_vals, degrees, indexing="ij")

    cf = ax.pcolormesh(
        ds,
        ls,
        data,
        shading="nearest",
        cmap=cmap_mse,
        vmin=0,
        vmax=0.1,
    )

    fig.colorbar(cf)

    ax.set_yscale("log")
    ax.set_ylabel(r"$\lambda$")
    ax.set_xlabel("Polynomial degree")
    ax.set_title("MSE")

    # title & save
    fig.suptitle(f"Polynomial Ridge Regression on $n={x_train.shape[0]}$ points")
    fig.savefig(os.path.join(FIGURES_URL, "TESTTEST"))
    plt.close()
            
if __name__ == "__main__":
    Heatmap_vary_ridgeparam()
