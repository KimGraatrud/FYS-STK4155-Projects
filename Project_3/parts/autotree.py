import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src import utils
from .auto import Autoencoder, train


def train_models():


def train_tree():
    # TODO: this is example code for 


def main():
    rep_size = [8, 16, 32, 64]
    model_names = [f"auto_{s}.pt" for s in rep_size]

    models = [Autoencoder(s) for s in rep_size]

    for model, name in zip(models, model_names):
        path = os.path.join(utils.MODELS_URL, name)

        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()

    # Example code
    # Pick a model
    i = 2
    model = models[i]

    # Encode the training data
    train_feats, ytrain = FacesDataset(utils.DATA_URL, train=True).flat()
    test_feats, ytest = FacesDataset(utils.DATA_URL, train=False).flat()
    xtrain = model.encoder(torch.tensor(train_feats)).detach().numpy()
    xtest = model.encoder(torch.tensor(test_feats)).detach().numpy()

    print("xtrain", xtrain.shape)
    print("xtest", xtest.shape)
    print("ytrain", ytrain.shape)
    print("ytest", ytest.shape)
