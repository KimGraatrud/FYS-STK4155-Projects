import matplotlib.pyplot as plt
import numpy as np
from src import utils

plot_params = {
    'color': 'black',
    's': 50,
    'marker': 'D',
    # 'alpha': 0.6,
    'edgecolors': 'face',
}



def plot_pre_deeplearning_methods(show=False, save=True):


    # All of this data was taken from ../logs/tree.log

    dummy_tree = {
        'n': 'Dummy',       # Name
        'tr_RMSE': 0.5730,
        'tr_R2': 0.0000,
        'te_RMSE': 0.5646,
        'te_R2': -0.0001,
        'runtime': 1. / 60. # about one second
    }

    shallow_tree = {
        'n': 'Shallow Tree', # Name
        # 'tr_RMSE': 0.0000,
        # 'tr_R2': 1.0000,
        'te_RMSE': 0.3371,
        'te_R2': 0.6436,
        'runtime': 22.56    # Minutes
    }

    deep_tree = {
        'n': 'Deep Tree',   # Name
        'tr_RMSE': 0.0000,
        'tr_R2': 1.0000,
        'te_RMSE': 0.3744,
        'te_R2': 0.5603,
        'runtime': 126.68   # Minutes
    }

    gradBoost_OOB = {
        'n': 'GB',  # Name
        'tr_RMSE': 0.2218,
        'tr_R2': 0.8501,
        'te_RMSE': 0.2543,
        'te_R2': 0.7971,
        'runtime': 9.73         # Minutes
    }

    gradBoost_tuned = {
        'n': 'Tuned GB',  # Name
        # 'tr_RMSE': 0.2218,
        # 'tr_R2': 0.8501,
        'te_RMSE': 0.24531146374082655,
        'te_R2': 0.8111836626987035,
        'runtime': 9.73         # Minutes
    }

    # Test RMSE Plot
    fig, axs = plt.subplots(
        nrows=2,
        figsize=(utils.APS_COL_W, .9 * utils.APS_COL_W),
        sharex=True
    )
    fig.suptitle('Pre Deeplearning Model Preformance')

    ax = axs[0]
    ax.set_ylabel('R$^2$')
    ax.scatter(dummy_tree['n'], dummy_tree['te_R2'], **plot_params)
    ax.scatter(shallow_tree['n'], shallow_tree['te_R2'], **plot_params)
    ax.scatter(deep_tree['n'], deep_tree['te_R2'], **plot_params)
    ax.scatter(gradBoost_OOB['n'], gradBoost_OOB['te_R2'], **plot_params)
    ax.scatter(gradBoost_tuned['n'], gradBoost_tuned['te_R2'], **plot_params)

    
    ax = axs[1]
    ax.scatter(dummy_tree['n'], dummy_tree['te_RMSE'], **plot_params)
    ax.scatter(shallow_tree['n'], shallow_tree['te_RMSE'], **plot_params)
    ax.scatter(deep_tree['n'], deep_tree['te_RMSE'], **plot_params)
    ax.scatter(gradBoost_OOB['n'], gradBoost_OOB['te_RMSE'], **plot_params)
    ax.scatter(gradBoost_tuned['n'], gradBoost_tuned['te_RMSE'], **plot_params)

    ax.set_ylabel('Test RMSE')
    ax.tick_params(
        axis='x',
        labelrotation=-45,
        # rotation_mode='anchor',
    )
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('left')
    ax.set_ylim(0,.6)
    # fig.tight_layout()

    if save: plt.savefig(utils.FIGURES_URL+'test-RMSE-pre-deeplr.png')
    if show: plt.show()

def plot_post_deeplearning_methods(show=False, save=True):


    # All of this data was taken from ../logs/hybrid.log
    #   and CNNresults.log
    hybrid_GB = {
        'n': 'Hybrid GB',  # Name
        'tr_RMSE': 0.2218,
        'tr_R2': 0.8501,
        'te_RMSE': 0.2543,
        'te_R2': 0.7971,
        'combined runtime': 9.73         # Minutes
    }

    # Test RMSE Plot
    fig, axs = plt.subplots(
        nrows=2,
        figsize=(utils.APS_COL_W, .9 * utils.APS_COL_W),
        sharex=True
    )
    fig.suptitle('Pre Deeplearning Model Preformance')

    ax = axs[0]
    ax.set_ylabel('R$^2$')
    ax.scatter(dummy_tree['n'], dummy_tree['te_R2'], **plot_params)
    ax.scatter(shallow_tree['n'], shallow_tree['te_R2'], **plot_params)
    ax.scatter(deep_tree['n'], deep_tree['te_R2'], **plot_params)
    ax.scatter(gradBoost_OOB['n'], gradBoost_OOB['te_R2'], **plot_params)
    ax.scatter(gradBoost_tuned['n'], gradBoost_tuned['te_R2'], **plot_params)

    
    ax = axs[1]
    ax.scatter(dummy_tree['n'], dummy_tree['te_RMSE'], **plot_params)
    ax.scatter(shallow_tree['n'], shallow_tree['te_RMSE'], **plot_params)
    ax.scatter(deep_tree['n'], deep_tree['te_RMSE'], **plot_params)
    ax.scatter(gradBoost_OOB['n'], gradBoost_OOB['te_RMSE'], **plot_params)
    ax.scatter(gradBoost_tuned['n'], gradBoost_tuned['te_RMSE'], **plot_params)

    ax.set_ylabel('Test RMSE')
    ax.tick_params(
        axis='x',
        labelrotation=-45,
        # rotation_mode='anchor',
    )
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('left')
    ax.set_ylim(0,.6)
    # fig.tight_layout()

    if save: plt.savefig(utils.FIGURES_URL+'test-RMSE-pre-deeplr.png')
    if show: plt.show()



def main():

    plot_pre_deeplearning_methods()
    plot_post_deeplearning_methods()


if __name__=='__main__':
    main()