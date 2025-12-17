import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src import utils
from src.Dataset import GalaxyDataset


class Autoencoder(nn.Module):
    def __init__(self, laten_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, laten_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(laten_dim, 32 * 16 * 16),
            nn.Unflatten(1, (32, 16, 16)),
            nn.ConvTranspose2d(32, 32, 3, padding=1, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 5, 3, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(model):
    model.to(utils.device)

    dataset = GalaxyDataset(mode="train")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss()

    epochs = 10
    i = 0
    rolling_loss = 0

    for epoch in range(epochs):
        for batch in iter(dataloader):
            i += 1
            features, _ = batch
            features = features.to(utils.device)

            optimizer.zero_grad()

            # forward
            pred = model(features)
            loss = criterion(pred, features)

            # train
            loss.backward()
            optimizer.step()

            # reporting
            rolling_loss += loss.item()
            if i % 100 == 0:
                print(f"[{epoch}] {rolling_loss / i:.4f}")
                rolling_loss = 0.0
                i = 0

        print("epoch", epoch)


def plot_transition():
    model = Autoencoder()
    model.load_state_dict(torch.load("./models/auto.pt", weights_only=True))
    model.eval()

    dataset = GalaxyDataset(mode="train")
    num = 2
    loader = DataLoader(dataset, batch_size=num, shuffle=True)
    loader_it = iter(loader)
    start_batch, _ = next(loader_it)
    end_batch, _ = next(loader_it)

    for i in range(num):
        start = start_batch[i]
        end = end_batch[i]

        start_rep = model.encoder(start.unsqueeze(0)).detach()
        end_rep = model.encoder(end.unsqueeze(0)).detach()

        diff = end_rep - start_rep

        stages = 4

        fig, axs = plt.subplots(ncols=stages + 2)
        axs[0].imshow(start[0])
        axs[-1].imshow(end[0])

        for j in range(stages):
            rep = start_rep + (diff / stages) * j
            reprod = model.decoder(rep).detach()
            reprod = reprod.reshape((5, 64, 64))

            axs[j + 1].imshow(reprod[0])

        for ax in axs:
            ax.set_axis_off()

        fig.savefig(os.path.join(utils.FIGURES_URL, f"auto_{i}"))
        plt.close(fig)


def main():
    save = True
    model = Autoencoder()

    try:
        train(model)
    except KeyboardInterrupt:
        pass

    if save:
        torch.save(model.state_dict(), "./models/auto.pt")
