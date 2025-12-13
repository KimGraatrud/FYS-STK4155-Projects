import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from . import utils
from src.Dataset import GalaxyDataset


class CNN(nn.Module):
    def __init__(self, *args, kernal_size=3, actvn=nn.LeakyReLU, id=None):
        super(CNN, self).__init__()
        sizes = list(args)
        layers = []

        prev = 5
        while len(sizes) > 0:
            cur = sizes.pop(0)
            layers.append(
                nn.Conv2d(
                    prev,
                    cur,
                    kernel_size=kernal_size,
                    padding=1,
                )
            )
            layers.append(actvn())
            prev = cur

        layers.append(nn.MaxPool2d(4, 4))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(cur * 16 * 16, 1))

        self.network = nn.Sequential(*layers)
        self.id = id

    def filepath(self):
        if self.id is not None:
            return os.path.join(utils.MODELS_URL, f"{self.id}.pt")
        else:
            return None

    def forward(self, x):
        return self.network(x)


def train(model, epochs=10, device="cpu", batch_size=256):
    model.to(device)

    dataset = GalaxyDataset(mode="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss()

    i = 0
    rolling_loss = 0

    for epoch in range(epochs):
        for batch in iter(dataloader):
            i += 1
            features, labels = batch
            features = features.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            # forward
            pred = model(features).squeeze()

            loss = criterion(pred, labels)

            # train
            loss.backward()
            optimizer.step()

            # reporting
            rolling_loss += loss.item()
            if i % 100 == 0:
                print(f"[{epoch}] {rolling_loss / i:.4f}")
                rolling_loss = 0.0
                i = 0

    dataset.close()
