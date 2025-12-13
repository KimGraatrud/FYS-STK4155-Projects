import os
from os.path import join
import numpy as np
import torch

DATA_PATH = "./data/"
DATA_PATHS = {
    "train": join(DATA_PATH, "5x64x64_training_with_morphology.hdf5"),
    "validate": join(DATA_PATH, "5x64x64_validation_with_morphology.hdf5"),
    "test": join(DATA_PATH, "5x64x64_testing_with_morphology.hdf5"),
}
FIGURES_URL = "./figures/"
MODELS_URL = "./models/"
RESULTS_URL = "./results/"

# For figure sizing
APS_COL_W = 246 / 72.27  # (col width in pts / pts in inch)

# Seed for randomness
SEED = 2025
rng = np.random.default_rng(seed=SEED)

# DEVICE = torch.device("mps")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create a home for figures, models, etc.
def create_directories():
    if not os.path.exists(FIGURES_URL):
        os.mkdir(FIGURES_URL)

    if not os.path.exists(MODELS_URL):
        os.mkdir(MODELS_URL)

    if not os.path.exists(RESULTS_URL):
        os.mkdir(RESULTS_URL)


def trainable_params(model):
    ps = filter(lambda p: p.requires_grad, model.parameters())
    return np.sum([np.prod(p.size()) for p in ps])


def shuffle_idx(array):
    indicies = np.arange(len(array))
    rng.shuffle(indicies)
    return indicies


def print_tree_data(treeModel):

    print(f"{treeModel}'s stats:")
    print(f"    - Depth: {treeModel.get_n_leaves()}")
    print(f"    - Number of leaves: {treeModel.get_n_leaves()}")
    print(f"    - Parameters: {treeModel.get_params()}")
