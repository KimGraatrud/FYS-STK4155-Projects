import matplotlib.pyplot as plt
from parts import cnn_training, galaxies
from src import utils
import torch

torch.manual_seed(utils.SEED)

utils.create_directories()

plt.style.use("./style.mplstyle")

cnn_training.main()
# galaxies.issues()
# galaxies.demo()
# autotree.main()
# auto.main()
# auto.plot_transition()
