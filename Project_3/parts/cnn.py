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


class CNN(nn.Module):
    def __init__(self, *args, kernal_size=3, actvn=nn.LeakyReLU, id=None):
        super(CNN, self).__init__()
        sizes = list(args)
        layers = []

        prev = 5
        while len(sizes) > 0:
            cur = sizes.pop(0)
            layers.append(nn.Conv2d(prev, cur, 3, padding=1))
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


def _train(model, epochs=10, device="cpu", batch_size=256):
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


def _init_models():
    # 'deep' models
    d1 = CNN(8, 16, id="d1")
    d2 = CNN(8, 16, 16, id="d2")
    d3 = CNN(16, 16, 32, 32, id="d3")
    d4 = CNN(16, 16, 32, 32, 32, id="d4")
    d5 = CNN(16, 16, 32, 32, 64, 64, id="d5")
    d6 = CNN(16, 16, 16, 32, 32, 32, 64, 64, 64, id="d6")

    # 'wide' models
    w1 = CNN(8, kernal_size=5, id="w1")
    w2 = CNN(16, kernal_size=5, id="w2")
    w3 = CNN(16, 32, kernal_size=5, id="w3")
    w4 = CNN(16, 32, 64, kernal_size=5, id="w4")
    w5 = CNN(16, 32, 64, 128, kernal_size=5, id="w5")
    w6 = CNN(16, 32, 64, 128, 256, kernal_size=5, id="w6")

    ds = [d1, d2, d3, d4, d5, d6]
    ws = [w1, w2, w3, w4, w5, w6]

    return ds, ws


def train_models():
    # --------------------
    epochs = 1
    batch_size = 256
    # device = torch.device("mps")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --------------------

    print("device", device)

    ds, ws = _init_models()

    print("Number of trainable params:")
    print("wide deep")
    for w, d in zip(ws, ds):
        print(utils.trainable_params(w), utils.trainable_params(d))

    for model in [*ds, *ws]:
        print(model.id)
        _train(model, epochs=epochs, device=device, batch_size=batch_size)

    for model in [*ds, *ws]:
        torch.save(model.state_dict(), model.filepath())


def evaluate_models(mode="validate"):
    # device = torch.device("mps")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --------------------

    ds, ws = _init_models()

    for model in [*ds, *ws]:
        model.load_state_dict(torch.load(model.filepath(), weights_only=True))

    dataset = GalaxyDataset(mode=mode)
    ds_loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    for models in [ds, ws]:
        for i, model in enumerate(models):
            print(model.id)

            # Reporting & Plotting
            fig, ax = plt.subplots(
                figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W),
            )

            preds = torch.tensor([])
            with torch.no_grad():
                model.to(device)
                for imgs, _ in ds_loader:
                    imgs = imgs.to(device)
                    pred = model(imgs).squeeze().cpu()
                    preds = torch.cat((preds, pred))

            # reference line
            ax.plot(np.linspace(0, 4, 30), np.linspace(0, 4, 30), c="k", lw=1)

            ax.hist2d(
                dataset.z,
                preds.numpy(),
                bins=300,  # increase this if you have memory to spare
                range=[[0, 4], [0, 4]],
                norm="log",
            )

            ax.set_ylim(0, 4)
            ax.set_title(model.id)
            if i == 0:
                ax.set_xlim(0, 4)

            fig.savefig(os.path.join(utils.FIGURES_URL, f"zz_{model.id}"))
            plt.close(fig)


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
