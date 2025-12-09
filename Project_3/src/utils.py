import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


DATA_URL = "./Data/"
APS_COL_W = 246 / 72.27  # (col width in pts / pts in inch)
FIGURES_URL = "./figures/"
SEED = 2025

# create a home for figures
if not os.path.exists(FIGURES_URL):
    os.mkdir(FIGURES_URL)


def dataset_accuracy(model, dataset):
    # First make sure the model is on the cpu
    cpu = torch.device("cpu")
    model.to(cpu)

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    correct = 0
    total = 0

    for batch, labels in iter(loader):
        predict = model(batch)
        _, idxs = torch.max(predict, 1)

        correct += np.equal(idxs, labels).sum()
        total += len(labels)

    return correct / total


def dataset_accuracy_breakdown(model, dataset):
    # First make sure the model is on the cpu
    cpu = torch.device("cpu")
    model.to(cpu)

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    correct = np.zeros(5)
    total = np.zeros_like(correct)

    for batch, labels in iter(loader):
        # TODO: this should be faster
        predict = model(batch)
        _, idxs = torch.max(predict, 1)

        for i, idx in enumerate(idxs):
            cat = labels[i]
            correct[cat] += idx == cat
            total[cat] += 1

    return correct, total


def error_rate(target, pred):
    return 1 - accuracy_score(target, pred)
