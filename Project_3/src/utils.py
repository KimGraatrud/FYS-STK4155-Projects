import numpy as np
import torch
from torch.utils.data import DataLoader

DATA_URL = "./Data/"


def report_dataset_accuracy(model, dataset):
    # First make sure the model is on the cpu
    cpu = torch.device("cpu")
    model.to(cpu)

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    correct = 0.0
    total = 0.0

    # WORKING ON, but leaving out for right now so that
    # I can make kim a plot
    # -S

    # by_label_correct = {}
    # by_label_total = {}

    for batch, labels in iter(loader):
        predict = model(batch)
        _, idxs = torch.max(predict, 1)

        correct += np.equal(idxs, labels).sum()
        total += len(labels)

    return correct / total
