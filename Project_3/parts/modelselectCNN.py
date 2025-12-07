import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src import utils
from src.FacesDataset import FacesDataset
from joblib import Parallel, delayed
import itertools
import time

# Temporary class placement
class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()

        self.c1 = nn.Conv2d(1, 4, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(4, 8, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.l1 = nn.Linear(8 * 9 * 9, 64)
        self.l2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.pool1(x)
        x = F.relu(self.c2(x))
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
# End of temporary placement

def train_and_eval(params, trainset, testset, device):
    lr = params["lr"]
    momentum = params["momentum"]
    batch_size = params["batch_size"]

    # dataloaders for this run
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(0)  # reproducible

    model = Machine().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = params.get("epochs", 5)

    for epoch in range(EPOCHS):
        model.train()
        for features, labels in trainloader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    # evaluate
    acc = utils.dataset_accuracy(model, testset).item()
    return params, acc

def gridsearch(trainset, testset, reg_grid,
                weight_grid, batch_grid,
                epoch_grid, device):
    """
    Grid search over the parameter space.

    Runs over all relevant parameters.

    Multithreaded using joblib.
    """

    # Define the parameters
    param_grid = {
        "lr": reg_grid,
        "momentum": weight_grid,
        "batch_size": batch_grid,
        "epochs": epoch_grid,
    }

    # Build all combinations
    keys, values = zip(*param_grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Parallel evaluation
    searchstart = time.time()
    results = Parallel(n_jobs=1)(
        delayed(train_and_eval)(p, trainset, testset, device)
        for p in combos
    )
    searchend = time.time()

    print(f'Gridsearch finished after: {searchend-searchstart} seconds.')

    # Pick best
    best_params, best_acc = max(results, key=lambda x: x[1])

    print("Best parameters:", best_params)
    print("Best accuracy:", best_acc)

    return best_params, best_acc

def main():

    # Get data
    trainset = FacesDataset(utils.DATA_URL, train=True)
    # trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = FacesDataset(utils.DATA_URL, train=False)
    # testloader = DataLoader(testset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # Define the grid
    etas = np.logspace(-4, -1, 10)
    weights = np.linspace(0, 1, 10)
    baches = np.arange(64, 1024, 64)
    eopcs = np.array([5, 10])

    # Do the grid search
    best_params, best_accuracy = gridsearch(
        trainset, testset, L2s, weights, baches, eopcs, device
    )


