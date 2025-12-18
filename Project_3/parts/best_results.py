import matplotlib.pyplot as plt
import numpy as np
from src import utils
from .cnn_plotting import eval_rmse_r2

plot_params = {
    'color': 'black',
    's': 50,
    'marker': 'D',
    # 'alpha': 0.6,
    'edgecolors': 'face',
}

    # All of this data was taken from ../logs/tree.log

dummy_tree = {
    'n': 'Dummy',                           # Name
    'tr_RMSE': 0.5729750604933044,          # RMSE
    'tr_R2': 0.0,                           # R2
    'te_RMSE': 0.5645691368252176,          # RMSE
    'te_R2': -8.784078060997125e-05,        # R2
    'tr_runtime': 0.0007107615470886231,    # Minutes
    'pred_runtime': 2.6941299438476564e-06  # Minutes
}

shallow_tree = {
    'n': 'Shallow Tree', 
    'tr_RMSE': 0.33815047599496917,
    'tr_R2': 0.6517039349691438,
    'te_RMSE': 0.3370503004469756,
    'te_R2': 0.643554350563769,
    'tr_runtime': 23.387147760391237,
    'pred_runtime': 0.008310397466023764
}

deep_tree = {
    'n': 'Deep Tree',   
    'tr_RMSE': 0.0000,
    'tr_R2': 1.0000,
    'te_RMSE': 0.3744,
    'te_R2': 0.5603,
    'tr_runtime': 124.94320247173309,
    'pred_runtime': 0.007022507985432943
}

gradBoost_OOB = {
    'n': 'GB',
    'tr_RMSE': 0.22180729601365803,
    'tr_R2': 0.8501419018203292,
    'te_RMSE': 0.2543033728552935,
    'te_R2': 0.7970878176675769,
    'tr_runtime': 6.582000597318014,
    'pred_runtime': 0.07716052532196045
}

gradBoost_tuned = {
    'n': 'Tuned GB',
    'tr_RMSE': 0.18817331002635287,
    'tr_R2': 0.8921439253353459,
    'te_RMSE': 0.24935609109189016,
    'te_R2': 0.8049060311600303,
    'tr_runtime': 9.047373974323273,
    'pred_runtime': 0.10793048540751139
}
hybrid_GB = {
    'n': 'Hybrid GB',
    'tr_RMSE': 0.2227782730706814,
    'tr_R2': 0.8501,
    'te_RMSE': 0.24366036233090982,
    'te_R2': 0.8137168159564558,
    'tr_runtime': 152.63464641571045/60 + 0.5543735027313232,      # CNN training time + tree training time
    'pred_runtime': 12.063827276229858/60 + 0.18664073944091797         # feature extraction time + tree prediction time
}


def plot_tree_preformance(show=False, save=True):

    models = [
        dummy_tree,
        shallow_tree,
        deep_tree,
        gradBoost_OOB,
        gradBoost_tuned,
        hybrid_GB
    ]

    # Test RMSE Plot
    fig, axs = plt.subplots(
        nrows=2,
        figsize=(utils.APS_COL_W, 1.2 * utils.APS_COL_W),
        sharex=True
    )
    fig.suptitle('Pre Deeplearning Model Preformance')
    
    for ax in axs:
        ax.grid(True, alpha=.6)

    for model in models:
        axs[0].scatter(model['n'], model['te_R2'], **plot_params)
        axs[1].scatter(model['n'], model['tr_runtime'], **plot_params)

    axs[1].set_yscale('log')
    axs[0].set_ylabel('R$^2$')
    axs[1].set_ylabel('Training time (minutes)')
    axs[1].tick_params(
        axis='x',
        labelrotation=-45,
        # rotation_mode='anchor',
    )
    for label in axs[1].get_xticklabels():
        label.set_horizontalalignment('left')
    axs[0].set_ylim(None, 1)
    axs[1].set_ylim(10**0, None)
    fig.tight_layout()
    

    if save: plt.savefig(utils.FIGURES_URL+'test-RMSE-pre-deeplr.png')
    if show: plt.show()

def plot_post_deeplearning_methods(show=False, save=True):

    models = [
        dummy_tree,
        shallow_tree,
        deep_tree,
        gradBoost_OOB,
        gradBoost_tuned,
        hybrid_GB
    ]



    # Scaled Preformance Plot
    fig, axs = plt.subplots(
        nrows=2,
        figsize=(utils.APS_COL_W, 1.2 * utils.APS_COL_W),
        sharex=True
    )
    fig.suptitle('Pre Deeplearning Model Preformance')
    
    for ax in axs:
        ax.grid(True, alpha=.6)

    # Scale their preformance
    # And plot
    for model in models:
        axs[0].scatter(model['n'], model['te_R2'], **plot_params)
        axs[1].scatter(model['n'], model['te_RMSE'], **plot_params)

    axs[0].set_ylabel('R$^2$')
    axs[1].set_ylabel('Training time (minutes)')
    axs[1].tick_params(
        axis='x',
        labelrotation=-45,
        # rotation_mode='anchor',
    )
    for label in axs[1].get_xticklabels():
        label.set_horizontalalignment('left')
    axs[0].set_ylim(None, .90)
    fig.tight_layout()



    if save: plt.savefig(utils.FIGURES_URL+'test-RMSE-pre-deeplr.png')
    if show: plt.show()



def main():

    plot_tree_preformance()
    # plot_post_deeplearning_methods()

    print(hybrid_GB['tr_runtime'])
    print(hybrid_GB['pred_runtime'])


if __name__=='__main__':
    main()