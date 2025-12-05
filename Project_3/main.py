import pickle
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

torch.manual_seed(0)

trainset = FacesDataset(utils.DATA_URL, train=True)
print("trainset", len(trainset))
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = FacesDataset(utils.DATA_URL, train=False)
print("testset", len(testset))
testloader = DataLoader(testset, batch_size=64, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")
print("device", device)


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


m = Machine()
m.to(device)

optimizer = optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 10
try:
    for epoch in range(epochs):
        rolling_loss = 0
        i = 0
        for batch in iter(trainloader):
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
                print(f"[{epoch} {i}] {rolling_loss / 100:.3f}")
                rolling_loss = 0.0

            i += 1
except KeyboardInterrupt:
    print("Calculating accuracy:")
    pass

train_acc = utils.dataset_accuracy(m, trainset)
print("train_acc", train_acc.item())
test_acc = utils.dataset_accuracy(m, testset)
print("test_acc", test_acc.item())
