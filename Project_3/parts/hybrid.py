from src import utils, Dataset
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler  # Feature scaling after
from sklearn.model_selection import train_test_split

from .cnn_training import init_model

# set torch seed
torch.manual_seed(utils.SEED)


class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()

        self.c1 = nn.Conv2d(5, 8, 3, padding=1)
        self.pool1 = nn.AvgPool2d(3, 3)
        self.c2 = nn.Conv2d(8, 8, 5, padding=2)
        self.pool2 = nn.AvgPool2d(3, 3)
        self.l1 = nn.Linear(8 * 3 * 3, 5)

    def conv_forward(self, x):
        x = F.leaky_relu(self.c1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.c2(x))
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x):

        x = self.conv_forward(x)
        return self.l1(x)


def calc_acc(data, model):

    trainset, testset = data
    m = model

    print("\nCalculating accuracy:")

    print("Train", utils.dataset_accuracy(m, trainset))
    print("Test", utils.dataset_accuracy(m, testset))

    train_correct, train_total = utils.dataset_accuracy_breakdown(m, trainset)
    test_correct, test_total = utils.dataset_accuracy_breakdown(m, testset)

    print(f"{'':6}{'Train':>10}{'Test':>10}")
    for i in range(5):
        print(
            f"{i:6}{train_correct[i] / train_total[i]:10.4f}{test_correct[i] / test_total[i]:10.4f}"
        )

    print(
        f"{'Total':>6}{np.sum(train_correct) / np.sum(train_total):10.4f}{np.sum(test_correct) / np.sum(test_total):10.4f}"
    )


def trainmodel(trainset, device, max_epocs=50, batchsize=32, verbose=False):
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
    m = Machine()

    optimizer = optim.Adam(m.parameters(), lr=0.05, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    print("\ntraining")
    # Training loop
    try:
        for epoch in range(max_epocs):
            rolling_loss = 0
            i = 0
            for features, labels in iter(trainloader):
                i += 1

                # move tensors to the device
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
                if verbose:
                    if i % 100 == 99:
                        print(f"[{epoch}], Loss: {rolling_loss / i:.3f}")
                        rolling_loss = 0.0
                        i = 0

            print("epoch", epoch, rolling_loss / i)

    except KeyboardInterrupt:
        pass

    return m


def extract_conv_features(dataset, model, device, batchsize=32):
    """
    Extract features of the dataset after training.
    Also return its labels, and convert both to numpy.
    """

    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
    all_features = []
    all_labels = []

    # Set model in eval mode
    model.eval()

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            feats = model.conv_forward(x)
            all_features.append(feats.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_features, all_labels


def main():

    device = utils.device
    print("Using device:", device)

    trainset = Dataset.GalaxyDataset("train")
    testset = Dataset.GalaxyDataset("test")
    print("trainset", len(trainset))
    print("testset", len(testset))

    model = trainmodel(trainset, device=device, verbose=True, max_epocs=10)
    tr_features, tr_labels = extract_conv_features(trainset, model, device=device)
    te_features, te_labels = extract_conv_features(testset, model, device=device)

    # Find out how much it improved the normal tree preformance
    model = DecisionTreeClassifier().fit(tr_features, tr_labels)
    normalTreetrain = model.predict(tr_features)
    normalTreetest = model.predict(te_features)

    print(
        "normal tree RMSE train",
        np.sqrt(mean_squared_error(tr_labels, normalTreetrain)),
    )
    print(
        "normal tree RMSE test", np.sqrt(mean_squared_error(te_labels, normalTreetest))
    )
    print("normal tree R^2 test", r2_score(te_labels, normalTreetest))

    utils.print_tree_data(model)
