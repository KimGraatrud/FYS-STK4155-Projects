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
from src import utils
from src.FacesDataset import FacesDataset

torch.manual_seed(0)


class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()

        self.c1 = nn.Conv2d(1, 8, 5)
        self.c2 = nn.Conv2d(8, 16, 5)
        self.pool1 = nn.AvgPool2d(4, 4)
        # self.c3 = nn.Conv2d(32, 64, 3)
        # self.pool2 = nn.MaxPool2d(3, 3)

        self.l1 = nn.Linear(16 * 10 * 10, 5)
        # self.l2 = nn.Linear(16, 5)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.pool1(x)
        # x = F.relu(self.c3(x))
        # x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)

        return self.l1(x)
        # x = F.relu(self.l1(x))
        # x = self.l2(x)
        # return x


def main():

    trainset = FacesDataset(utils.DATA_URL, train=True)
    testset = FacesDataset(utils.DATA_URL, train=False)
    print("trainset", len(trainset))
    print("testset", len(testset))

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    m = Machine()

    # load = False
    load = True
    if load:
        print("Loading model")
        m.load_state_dict(torch.load("./models/model.pt", weights_only=True))
        m.eval()

    else:
        print("Training model")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # device = torch.device("cpu")
        print("device", device)

        m.to(device)

        # report number of params
        print("Trainable params per layer:")
        total = 0
        for p in m.parameters():
            s = p.size().numel()
            total += s
            print(s)
        print("total:", total)

        optimizer = optim.Adam(m.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.CrossEntropyLoss()

        print("\ntraining")
        # Training loop
        epochs = 100
        try:
            for epoch in range(epochs):
                rolling_loss = 0
                i = 0
                for batch in iter(trainloader):
                    i += 1
                    features, labels = batch

                    # move tensors to the GPU if using
                    features = features.to(device)
                    labels = labels.to(device)

                    m.zero_grad()
                    optimizer.zero_grad()

                    result = m(features)
                    loss = criterion(result, labels)

                    l = loss.item()

                    if l != l:
                        print("nan encountered!")
                        break

                    loss.backward()
                    optimizer.step()

                    rolling_loss += loss.item()
                    if i % 100 == 99:
                        print(f"[{epoch}] {rolling_loss / i:.3f}")
                        rolling_loss = 0.0
                        i = 0

                print("epoch", epoch, rolling_loss / i)

        except KeyboardInterrupt:
            pass

        save = False
        # save = True
        if save:
            print("Saving model")
            torch.save(m.state_dict(), "./models/model.pt")

    ex_image = torch.unsqueeze(decode_image("./Data/Happy/537.png"), 0) / 255.0
    # print(ex_image)

    l1 = F.relu(m.c2(F.relu(m.c1(ex_image))))

    channels = l1.detach()[0]
    fig, axs = plt.subplots(len(channels))
    for i, channel in enumerate(channels):
        axs[i].imshow(channel)

    fig.savefig(os.path.join(utils.FIGURES_URL, "test.png"))
    plt.close(fig)

    # print("\nCalculating accuracy:")

    # print("Train", utils.dataset_accuracy(m, trainset))
    # print("Test", utils.dataset_accuracy(m, testset))

    # train_correct, train_total = utils.dataset_accuracy_breakdown(m, trainset)
    # test_correct, test_total = utils.dataset_accuracy_breakdown(m, testset)

    # print(f"{'':6}{'Train':>10}{'Test':>10}")
    # for i in range(5):
    #     print(
    #         f"{i:6}{train_correct[i] / train_total[i]:10.4f}{test_correct[i] / test_total[i]:10.4f}"
    #     )

    # print(
    #     f"{'Total':>6}{np.sum(train_correct) / np.sum(train_total):10.4f}{np.sum(test_correct) / np.sum(test_total):10.4f}"
    # )
