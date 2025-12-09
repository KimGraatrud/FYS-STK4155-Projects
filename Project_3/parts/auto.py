import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import src
from src import utils
from src.FacesDataset import FacesDataset


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.l1 = nn.Linear(48 * 48, 512)
        self.mu = nn.Linear(512, 128)
        self.sigma = nn.Linear(512, 128)
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, x):
        y = F.relu(self.l1(x))
        mu = self.mu(y)
        sigma = F.sigmoid(self.sigma(y))

        z = mu + sigma * self.N.sample(sigma.shape)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.l1 = nn.Linear(128, 512)
        self.l2 = nn.Linear(512, 48 * 48)

    def forward(self, x):
        y = F.relu(self.l1(x))
        z = torch.sigmoid(self.l2(y))
        return z


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z


def train(model):
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print("device", device)
    model.to(device)

    dataset = FacesDataset(utils.DATA_URL)
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
            features = features.flatten(start_dim=1)
            features = features.to(device)

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

    for train in [True, False]:
        dataset = FacesDataset(utils.DATA_URL, train=train)
        num = 4
        loader = DataLoader(dataset, batch_size=num, shuffle=True)
        loader_it = iter(loader)
        start_batch, _ = next(loader_it)
        end_batch, _ = next(loader_it)

        for i in range(num):
            start = start_batch[i]
            end = end_batch[i]

            start_rep = model.encoder(start.flatten(start_dim=1)).detach()
            end_rep = model.encoder(end.flatten(start_dim=1)).detach()

            diff = end_rep - start_rep

            stages = 4

            fig, axs = plt.subplots(ncols=stages + 2)
            axs[0].imshow(start[0])
            axs[-1].imshow(end[0])

            for j in range(stages):
                rep = start_rep + (diff / stages) * j
                reprod = model.decoder(rep).detach()
                reprod = reprod.reshape((1, 48, 48))

                axs[j + 1].imshow(reprod[0])

            for ax in axs:
                ax.set_axis_off()

            fig.savefig(
                os.path.join(utils.FIGURES_URL, f"auto_{['test', 'train'][train]}_{i}")
            )
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
