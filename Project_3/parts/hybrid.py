from src import utils, Dataset
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from .cnn_training import _evaluate_all
from .cnn_plotting import eval_rmse_r2

# set torch seed
torch.manual_seed(utils.SEED)


class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()

        self.c1 = nn.Conv2d(5, 8, 3)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.c2 = nn.Conv2d(8, 8, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.c3 = nn.Conv2d(8, 8, 5)
        self.pool3 = nn.AvgPool2d(3, 3)
        self.l1 = nn.Linear(8 * 3 * 3, 1)

    def conv_forward(self, x):
        x = F.leaky_relu(self.c1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.c2(x))
        x = self.pool2(x)
        x = F.leaky_relu(self.c3(x))
        x = self.pool3(x)

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
    m.to(device)

    trainable_params = utils.trainable_params(m)
    print('Number of trainable params:', trainable_params)
    raise

    optimizer = optim.Adam(m.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss()

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
                labels = labels.float().to(device)

                m.zero_grad()
                optimizer.zero_grad()

                result = m(features).squeeze()
                loss = torch.sqrt(criterion(result, labels))

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

    # CNN Training
    CNN_train_start = time.time()
    CNNmodel = trainmodel(trainset, device=device, verbose=True, max_epocs=10)
    CNN_train_end = time.time()

    print("CNN training time (s):", CNN_train_end - CNN_train_start)

    torch.save(CNNmodel.state_dict(), utils.MODELS_URL + "hybridCNN")

    CNNmodel = Machine()
    state = torch.load(utils.MODELS_URL + "hybridCNN.pt", weights_only=True, map_location="cpu")
    CNNmodel.load_state_dict(state)

    CNNmodel.eval()

    bg = time.time()
    prediction = _evaluate_all(CNNmodel, testset, load=False)
    ed = time.time()
    print('pred took:', ed-bg)
    rmses, r2s = eval_rmse_r2(testset.z, prediction)
    print(rmses, r2s)

    # Feature Extraction
    feat_start = time.time()
    tr_features, tr_labels = extract_conv_features(trainset, CNNmodel, device=device)
    te_features, te_labels = extract_conv_features(testset, CNNmodel, device=device)
    feat_end = time.time()

    print("Feature extraction time (s):", feat_end - feat_start)
    print("Feature dimension:", tr_features.shape[1])

    # Regressor Tree
    tree_train_start = time.time()
    tree = DecisionTreeRegressor(
        random_state=utils.SEED
    )
    tree.fit(tr_features, tr_labels)
    tree_train_end = time.time()

    tree_pred_train_start = time.time()
    tree_pred_train = tree.predict(tr_features)
    tree_pred_test = tree.predict(te_features)
    tree_pred_train_end = time.time()

    print("\nDecision Tree results:")
    print("Training time (s):", tree_train_end - tree_train_start)
    print("Prediction time (s):", tree_pred_train_end - tree_pred_train_start)
    print("RMSE train:", np.sqrt(mean_squared_error(tr_labels, tree_pred_train)))
    print("RMSE test:", np.sqrt(mean_squared_error(te_labels, tree_pred_test)))
    print("R^2 test:", r2_score(te_labels, tree_pred_test))

    utils.print_tree_data(tree)

    # Gradient Boosting
    hgb_train_start = time.time()
    hgb = HistGradientBoostingRegressor(
        random_state=utils.SEED
    )
    hgb.fit(tr_features, tr_labels)
    hgb_train_end = time.time()

    hgb_pred_start = time.time()
    hgb_pred_train = hgb.predict(tr_features)
    hgb_pred_test = hgb.predict(te_features)
    hgb_pred_end = time.time()

    print("\nHistGradientBoosting results:")
    print("Training time (s):", hgb_train_end - hgb_train_start)
    print("Prediction time (s):", hgb_pred_end - hgb_pred_start)
    print("RMSE train:", np.sqrt(mean_squared_error(tr_labels, hgb_pred_train)))
    print("RMSE test:", np.sqrt(mean_squared_error(te_labels, hgb_pred_test)))
    print("R^2 test:", r2_score(te_labels, hgb_pred_test))