import matplotlib.pyplot as plt
from parts import cnn_training, cnn_plotting, galaxies
from src import utils
import torch

torch.manual_seed(utils.SEED)

utils.create_directories()

plt.style.use("./style.mplstyle")

print("device", utils.DEVICE)

cnn_training.main()
cnn_plotting.main()
# galaxies.issues()
# galaxies.demo()
# autotree.main()
# auto.main()
# auto.plot_transition()
