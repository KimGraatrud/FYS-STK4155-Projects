import os
from os.path import join
import numpy as np
import h5py as h5
import torch
from torch.utils.data import DataLoader
from src.Dataset import GalaxyDataset

DATA_PATH = "./data/"
DATA_PATHS = {
    "train": join(DATA_PATH, "5x64x64_training_with_morphology.hdf5"),
    "validate": join(DATA_PATH, "5x64x64_validation_with_morphology.hdf5"),
    "test": join(DATA_PATH, "5x64x64_testing_with_morphology.hdf5"),
}
FIGURES_URL = "./figures/"
MODELS_URL = "./models/"
RESULTS_URL = "./results/"
NORM_URL = join(RESULTS_URL, "norm.npz")

# For figure sizing
APS_COL_W = 246 / 72.27  # (col width in pts / pts in inch)

# Seed for randomness
SEED = 2025
rng = np.random.default_rng(seed=SEED)

# device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create a home for figures, models, etc.
def create_directories():
    if not os.path.exists(FIGURES_URL):
        os.mkdir(FIGURES_URL)

    if not os.path.exists(MODELS_URL):
        os.mkdir(MODELS_URL)

    best = os.path.join(MODELS_URL, "best")
    if not os.path.exists(best):
        os.mkdir(best)

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


def compute_normalization():
    print("computing normalization")
    ds = GalaxyDataset(mode="train", normalize=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    # This would prabably be faster if everything was
    # in memory at once, but simon's computer can't
    # handle that so we batch instead

    # mean
    total = np.zeros(ds.images.shape[1])
    for images, _ in loader:
        total += torch.sum(images, (0, 2, 3)).numpy()

    dss = ds.images.shape
    mean = total / (dss[0] * dss[2] * dss[3])
    print("mean:", mean)

    mean_exp = torch.tensor(np.expand_dims(mean, (0, 2, 3)), dtype=torch.float32)

    # std
    var = np.zeros_like(mean)
    for images, _ in loader:
        var += torch.sum((images - mean_exp) ** 2, (0, 2, 3)).numpy()

    std = np.sqrt(var / (dss[0] * dss[2] * dss[3] - 1))
    print("std:", std)

    ds.close()

    np.savez(NORM_URL, mean=mean, std=std)
