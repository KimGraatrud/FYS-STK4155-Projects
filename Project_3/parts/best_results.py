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
        'tr_RMSE': 0.5729750604933044,
        'tr_R2': 0.0,
        'te_RMSE': 0.5645691368252176,
        'te_R2': -8.784078060997125e-05,
        'tr_runtime': 0.0007107615470886231, # Minutes
        'pred_runtime': 2.6941299438476564e-06 # Minutes
    }

    shallow_tree = {
        'n': 'Shallow Tree', # Name
        'tr_RMSE': 0.33815047599496917,
        'tr_R2': 0.6517039349691438,
        'te_RMSE': 0.3370503004469756,
        'te_R2': 0.643554350563769,
        'tr_runtime': 23.387147760391237,    # Minutes
        'pred_runtime': 0.008310397466023764    # Minutes
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
        'tr_RMSE': 0.22180729601365803,
        'tr_R2': 0.8501419018203292,
        'te_RMSE': 0.2543033728552935,
        'te_R2': 0.7970878176675769,
        'tr_runtime': 6.582000597318014,         # Minutes
        'pred_runtime': 0.07716052532196045         # Minutes
    }

    gradBoost_tuned = {
        'n': 'Tuned GB',  # Name
        'tr_RMSE': 0.18817331002635287,
        'tr_R2': 0.8921439253353459,
        'te_RMSE': 0.24935609109189016,
        'te_R2': 0.8049060311600303,
        'tr_runtime': 9.047373974323273,         # Minutes
        'pred_runtime': 0.10793048540751139         # Minutes
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

# def plot_post_deeplearning_methods(show=False, save=True):


#     # All of this data was taken from ../logs/hybrid.log
#     #   and CNNresults.log
#     hybrid_GB = {
#         'n': 'Hybrid GB',  # Name
#         'tr_RMSE': 0.2218,
#         'tr_R2': 0.8501,
#         'te_RMSE': 0.2543,
#         'te_R2': 0.7971,
#         'combined runtime': 9.73         # Minutes
#     }

#     # Test RMSE Plot
#     fig, axs = plt.subplots(
#         nrows=2,
#         figsize=(utils.APS_COL_W, .9 * utils.APS_COL_W),
#         sharex=True
#     )
#     fig.suptitle('Pre Deeplearning Model Preformance')

#     ax = axs[0]
#     ax.set_ylabel('R$^2$')
#     ax.scatter(dummy_tree['n'], dummy_tree['te_R2'], **plot_params)
#     ax.scatter(shallow_tree['n'], shallow_tree['te_R2'], **plot_params)
#     ax.scatter(deep_tree['n'], deep_tree['te_R2'], **plot_params)
#     ax.scatter(gradBoost_OOB['n'], gradBoost_OOB['te_R2'], **plot_params)
#     ax.scatter(gradBoost_tuned['n'], gradBoost_tuned['te_R2'], **plot_params)

    
#     ax = axs[1]
#     ax.scatter(dummy_tree['n'], dummy_tree['te_RMSE'], **plot_params)
#     ax.scatter(shallow_tree['n'], shallow_tree['te_RMSE'], **plot_params)
#     ax.scatter(deep_tree['n'], deep_tree['te_RMSE'], **plot_params)
#     ax.scatter(gradBoost_OOB['n'], gradBoost_OOB['te_RMSE'], **plot_params)
#     ax.scatter(gradBoost_tuned['n'], gradBoost_tuned['te_RMSE'], **plot_params)

#     ax.set_ylabel('Test RMSE')
#     ax.tick_params(
#         axis='x',
#         labelrotation=-45,
#         # rotation_mode='anchor',
#     )
#     for label in ax.get_xticklabels():
#         label.set_horizontalalignment('left')
#     ax.set_ylim(0,.6)
#     # fig.tight_layout()

#     if save: plt.savefig(utils.FIGURES_URL+'test-RMSE-pre-deeplr.png')
#     if show: plt.show()



def main():

    plot_pre_deeplearning_methods()
    # plot_post_deeplearning_methods()


if __name__=='__main__':
    main()