import matplotlib.pyplot as plt
import numpy as np
from parts import auto
from src.FacesDataset import FacesDataset
from torch.utils.data import DataLoader
from src import utils

training, training_labels = FacesDataset(utils.DATA_URL).flat()


plt.style.use("./style.mplstyle")
auto.main()
auto.plot_transition()
# channels.main()
