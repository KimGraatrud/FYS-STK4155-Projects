import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from .. import utils, regression, resampling
from .A import create_data


def compare_bootstrap_kfold(x,y,maxdeg, lamR, lamL, nbootstraps, k):

    resampling_methods = resampling.resampling_methods(x,y,maxdeg, lamR, lamL)

    degrees, BOLS_stats, BRidge_stats, BLasso_stats = resampling_methods.BootstrapALL(
        nbootstraps, verbose=False
    )

    KOLS_MSE, KRidge_MSE, KLasso_MSE, KRidge_param, KLasso_param = resampling_methods.k_fold_CV(
        k, verbose=False
    )

    

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    colors = utils.colors

    ax.set_xlabel("Degree")
    ax.set_ylabel("MSE")

    ax.set_title("MSE comparison between Bootstrap and k-fold CV")

    ax.plot(degrees, BOLS_stats['MSE'], label='Bootstrap OLS', c=colors['ols'])
    ax.plot(degrees, KOLS_MSE, label='k-fold CV OLS', linestyle='dashed', c=colors['ols'])

    ax.plot(degrees, BRidge_stats['MSE'], label='Bootstrap OLS', c=colors['ridge'])
    ax.plot(degrees, KRidge_MSE, label='k-fold CV OLS', linestyle='dashed', c=colors['ridge'])

    ax.plot(degrees, BLasso_stats['MSE'], label='Bootstrap OLS', c=colors['lasso'])
    ax.plot(degrees, KLasso_MSE, label='k-fold CV OLS', linestyle='dashed', c=colors['lasso'])

    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.set_figheight(0.9 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "bootstrap-VS-kfold"))
    plt.close()

    


    

def main():

    n = 1e4
    n_bootstrap = 1000
    k = 10
    maxdegree = 10
    x, y = create_data(n)

    nlambdas = 100
    lamb_R = np.logspace(-5, -1, nlambdas)
    lamb_L = np.logspace(-4, 4, nlambdas)
    
    compare_bootstrap_kfold(x,y, maxdegree, lamb_R, lamb_L, n_bootstrap, k)



if __name__ == '__main__':
    main()

    
