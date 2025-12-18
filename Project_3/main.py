import os
import matplotlib.pyplot as plt
from parts import cnn_training, cnn_plotting, galaxies, auto
from src import utils
import torch

torch.manual_seed(utils.SEED)

utils.create_directories()

if not os.path.exists(utils.NORM_URL):
    utils.compute_normalization()


plt.style.use("./style.mplstyle")

print("device", utils.device)

cnn_training.main()
cnn_plotting.main()
# galaxies.issues()
# galaxies.demo()
# galaxies.test()
# autotree.main()
# auto.main()
# auto.plot_transition()
