import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from .. import utils, regression, resampling
from .A import create_data

def compare_bootstrap_kfold(x,y,maxdeg, lamR, lamL, nbootstraps, k):
    """
    Runs bootstrap and k-fold-CV for OLS and plots the training and 
    testing MSE for both in the same plot. 
    """

    resampling_methods = resampling.resampling_methods(x,y,maxdeg, lamR, lamL)

    degrees, Boot_result = resampling_methods.BootstrapOLS(
        nbootstraps, verbose=False
    )

    BMSE_train, Bbias2_train, Bvar_train = Boot_result['Train'] 
    BMSE_test, Bbias2_test, Bvar_test = Boot_result['Test'] 

    KOLS_MSE_train, KOLS_MSE_test = resampling_methods.k_fold_CV_OLS(
        k, verbose=False
    )

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    colors = utils.colors

    ax.set_xlabel("Degree")
    ax.set_ylabel("MSE")

    ax.set_title("MSE vs complexity for \n Bootstrap and k-fold CV")

    ax.plot(degrees, BMSE_train, label='Bootstrap Train', c=colors['ols'])
    ax.plot(degrees, BMSE_test, label='Bootstrap Test', linestyle='dashed', c=colors['ols'])

    ax.semilogy(degrees, KOLS_MSE_train, label='k-fold CV Train', c=colors['ridge'])
    ax.semilogy(degrees, KOLS_MSE_test, label='k-fold CV Test', linestyle='dashed', c=colors['ridge'])

    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.set_figheight(0.9 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "bootstrap-VS-kfold-MSE"))
    plt.close()


def make_k_fold_all_methods(x,y,maxdeg, lamR, lamL, k):

    resamp = resampling.resampling_methods(x,y,maxdeg, lamR, lamL)

    degrees = resamp.degrees

    OLS_MSE, Ridge_MSE, Lasso_MSE, Ridge_param, Lasso_param = resamp.k_fold_CV_ALL(
        k, verbose=True
    )

    np.savez(
        os.path.join(utils.DATA_URL, 'kfoldDATA.npz'),
        degrees=degrees,
        OLSMSE=OLS_MSE,
        RMSE=Ridge_MSE,
        LMSE=Lasso_MSE,
        Rparam=Ridge_param,
        Lparam=Lasso_param
    )

def load_k_fold_all_methods():

    a = np.load(
        os.path.join(utils.DATA_URL, 'kfoldDATA.npz'),
    )

    degrees = a['degrees']
    OLS_MSE = a['OLSMSE']
    Ridge_MSE = a['RMSE']
    Lasso_MSE = a['LMSE']
    Ridge_param = a['Rparam']
    Lasso_param = a['Lparam']

    return degrees, OLS_MSE, Ridge_MSE, Lasso_MSE, Ridge_param, Lasso_param

def plot_kfold_all(degrees, OLS_MSE, Ridge_MSE, Lasso_MSE):

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

    colors = utils.colors

    ax.set_xlabel("Degree")
    ax.set_ylabel("MSE")

    ax.set_title("MSE vs complexity for \n k-fold CV")

    ax.plot(degrees, OLS_MSE, label='OLS', c=colors['ols'])

    ax.plot(degrees, Ridge_MSE, label='Ridge', c=colors['ridge'])

    ax.plot(degrees, Lasso_MSE, label='Lasso', c=colors['lasso'])

    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.set_figheight(0.9 * utils.APS_COL_W)
    fig.savefig(os.path.join(utils.FIGURES_URL, "kfoldMSEALL"))
    plt.close()



def main():

    n = 1e2
    n_bootstrap = 1000
    k = 10
    maxdegree = 30
    x, y = create_data(n)

    nlambdas = 30
    lamb_R = np.logspace(-5, -1, nlambdas)
    lamb_L = np.logspace(-5, -1, nlambdas)
    
    # compare_bootstrap_kfold(x,y, maxdegree, lamb_R, lamb_L, n_bootstrap, k)

    make_k_fold_all_methods(x,y,maxdegree, lamb_R, lamb_L, k)
    degrees, OLS_MSE, Ridge_MSE, Lasso_MSE, Ridge_param, Lasso_param = load_k_fold_all_methods()
    plot_kfold_all(degrees, OLS_MSE, Ridge_MSE, Lasso_MSE)

    print(Ridge_param, Lasso_param)



if __name__ == '__main__':
    main()

    
