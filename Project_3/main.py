import matplotlib.pyplot as plt
from parts import cnn, galaxies

from src import utils
import torch

torch.manual_seed(0)

utils.create_directories()

# plt.style.use("./style.mplstyle")

galaxies.main()
cnn.train_models()
# cnn.small_demo()
# cnn.evaluate_models()
# autotree.main()
# auto.main()
# auto.plot_transition()
