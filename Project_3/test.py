import pickle
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src import utils, Dataset
from parts import tree, best_results, hybrid
from parts import tree, hybrid, best_results

plt.style.use("./style.mplstyle")

# print('Running tree.main()')
# tree.main()
print('Running hybrid.main()')
hybrid.main()
# print('Running best_results.main()')
# best_results.main()


