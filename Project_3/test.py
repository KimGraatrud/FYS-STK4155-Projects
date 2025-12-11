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
from parts import tree, modelselectCNN, hybrid

print('Running tree.main()')
tree.main()
# print('Running hybrid.main()')
# hybrid.main()