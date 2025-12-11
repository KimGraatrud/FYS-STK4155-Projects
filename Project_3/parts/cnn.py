import os
import pickle
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.io import decode_image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src import utils
from src.Dataset import GalaxyDataset


class Small(nn.Module):
    def __init__(self):
        super(Small, self).__init__()

        self.c1 = nn.Conv2d(5, 8, 3)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.c2 = nn.Conv2d(8, 8, 3)
        self.pool2 = nn.AvgPool2d(3, 3)
        self.l1 = nn.Linear(8 * 9 * 9, 1)

    def forward(self, x):
        x = F.leaky_relu(self.c1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.c2(x))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)

        return self.l1(x)


class Medium(nn.Module):
    def __init__(self):
        super(Medium, self).__init__()

        self.c1 = nn.Conv2d(5, 8, 3)
        self.c2 = nn.Conv2d(8, 16, 3)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.c3 = nn.Conv2d(16, 32, 4)
        self.pool2 = nn.AvgPool2d(3, 3)

        self.l1 = nn.Linear(32 * 9 * 9, 1)

    def forward(self, x):
        x = F.leaky_relu(self.c1(x))
        x = F.leaky_relu(self.c2(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.c3(x))
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)

        return self.l1(x)


class Large(nn.Module):
    def __init__(self):
        super(Large, self).__init__()

        self.c1 = nn.Conv2d(5, 8, 3)
        self.c2 = nn.Conv2d(8, 16, 3)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.c3 = nn.Conv2d(16, 32, 3)
        self.c4 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.AvgPool2d(3, 2)

        self.l1 = nn.Linear(64 * 11 * 11, 32)
        self.l2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.leaky_relu(self.c1(x))
        x = F.leaky_relu(self.c2(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.c3(x))
        x = F.leaky_relu(self.c4(x))
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)

        x = F.leaky_relu(self.l1(x))

        return self.l2(x)


def _train(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model.to(device)

    dataset = GalaxyDataset(mode="train")
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    epochs = 100
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

            # after computing val_loss
            scheduler.step(val_loss)

            # reporting
            rolling_loss += loss.item()
            if i % 100 == 0:
                print(f"[{epoch}] {rolling_loss / i:.4f}")
                rolling_loss = 0.0
                i = 0


def train_models():
    small = Small()
    medium = Medium()
    large = Large()

    try:
        _train(small)
        _train(medium)
        _train(large)
    except KeyboardInterrupt:
        pass

    torch.save(small.state_dict(), os.path.join(utils.MODELS_URL, "small.pt"))
    torch.save(medium.state_dict(), os.path.join(utils.MODELS_URL, "medium.pt"))
    torch.save(large.state_dict(), os.path.join(utils.MODELS_URL, "large.pt"))


def small_demo():
    # NOT FIXED YET
    raise
    small = Small()
    path = os.path.join(utils.MODELS_URL, "small.pt")
    small.load_state_dict(torch.load(path, weights_only=True))

    ex_image = torch.unsqueeze(decode_image("./Data/Happy/537.png"), 0) / 255.0

    x = ex_image

    fig = plt.figure(figsize=(utils.APS_COL_W, 0.6 * utils.APS_COL_W))
    gs = GridSpec(
        nrows=1, ncols=5, figure=fig, width_ratios=[1, 1, 1, 2, 2], wspace=0.1
    )
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.imshow(ex_image[0][0])
    ax1.set_axis_off()
    ax1.set_title("Original")

    # First layer
    x = F.leaky_relu(small.c1(x))
    xd = x.detach()
    subfig = fig.add_subfigure(gs[0, 1])
    subfig.set_facecolor("0.95")
    subfig.suptitle(r"$\mathrm{C}_1$")
    subgs = GridSpec(nrows=4, ncols=1, figure=subfig)
    for i, channel in enumerate(xd[0]):
        ax = subfig.add_subplot(subgs[i, 0])
        ax.set_axis_off()
        ax.imshow(channel)

    # pool
    x = small.pool1(x)
    xd = x.detach()
    subfig = fig.add_subfigure(gs[0, 2])
    subfig.suptitle(r"$\mathrm{P}_1$")
    subgs = GridSpec(nrows=4, ncols=1, figure=subfig)
    for i, channel in enumerate(xd[0]):
        ax = subfig.add_subplot(subgs[i, 0])
        ax.set_axis_off()
        ax.imshow(channel)

    # Second layer
    x = F.leaky_relu(small.c2(x))
    xd = x.detach()
    subfig = fig.add_subfigure(gs[0, 3])
    subfig.set_facecolor("0.95")
    subfig.suptitle(r"$\mathrm{C}_{2}$")
    subgs = GridSpec(nrows=4, ncols=2, figure=subfig)
    for i, channel in enumerate(xd[0]):
        row = int(np.floor(i / 2))
        col = i % 2
        ax = subfig.add_subplot(subgs[row, col])
        ax.set_axis_off()
        ax.imshow(channel)

    # pool
    x = small.pool2(x)
    xd = x.detach()
    subfig = fig.add_subfigure(gs[0, 4])
    subfig.suptitle(r"$\mathrm{P}_2$")
    subgs = GridSpec(nrows=4, ncols=2, figure=subfig)
    for i, channel in enumerate(xd[0]):
        row = int(np.floor(i / 2))
        col = i % 2
        ax = subfig.add_subplot(subgs[row, col])
        ax.set_axis_off()
        ax.imshow(channel)

    fig.savefig(os.path.join(utils.FIGURES_URL, "small_demo"))

    plt.close(fig)


def evaluate_models():
    # NOT FIXED YET
    raise
    models = [Small(), Medium(), Large()]
    names = ["small", "medium", "large"]
    testset = FacesDataset(utils.DATA_URL, train=False)

    for model, name in zip(models, names):
        model.load_state_dict(
            torch.load(os.path.join(utils.MODELS_URL, f"{name}.pt"), weights_only=True)
        )

        model.eval()

        print(name)
        correct, total = utils.dataset_accuracy_breakdown(model, testset)
        for i in range(5):
            print(f"{i:6}{correct[i] / total[i]:10.4f}")
        print(f"{'Total':>6}{np.sum(correct) / np.sum(total):10.4f}")
