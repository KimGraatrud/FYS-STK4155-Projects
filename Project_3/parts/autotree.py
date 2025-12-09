import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src import utils
from src.FacesDataset import FacesDataset


class Encoder(nn.Module):
    def __init__(self, small):
        super(Encoder, self).__init__()

        self.l1 = nn.Linear(48 * 48, 512)
        self.mu = nn.Linear(512, small)
        self.sigma = nn.Linear(512, small)
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, x):
        y = F.relu(self.l1(x))
        mu = self.mu(y)
        sigma = F.sigmoid(self.sigma(y))

        z = mu + sigma * self.N.sample(sigma.shape)
        return z


class Decoder(nn.Module):
    def __init__(self, small):
        super(Decoder, self).__init__()

        self.l1 = nn.Linear(small, 512)
        self.l2 = nn.Linear(512, 48 * 48)

    def forward(self, x):
        y = F.relu(self.l1(x))
        z = torch.sigmoid(self.l2(y))
        return z


class Autoencoder(nn.Module):
    def __init__(self, small):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(small)
        self.decoder = Decoder(small)

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z


def train(model):
    dataset = FacesDataset(utils.DATA_URL)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss()

    epochs = 10

    for epoch in range(epochs):
        i = 0
        rolling_loss = 0
        for batch in iter(dataloader):
            features, _ = batch
            features = features.flatten(start_dim=1)

            optimizer.zero_grad()

            # forward
            pred = model(features)
            loss = criterion(pred, features)

            # train
            loss.backward()
            optimizer.step()

            # reporting
            rolling_loss += loss.item()
            i += 1

        print(f"[{epoch}] {rolling_loss / i:.4f}")


def main():
    rep_size = [8, 16, 32, 64]
    model_names = [f"auto_{s}.pt" for s in rep_size]

    # Set to False to load models
    train_new = False
    # train_new = True

    models = [Autoencoder(s) for s in rep_size]

    for model, name in zip(models, model_names):
        path = os.path.join(utils.MODELS_URL, name)

        if train_new:
            print(f"Training {name}")
            train(model)

            torch.save(model.state_dict(), path)
        else:
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
