import matplotlib.pyplot as plt
from parts import cnn, galaxies
from src import utils
import torch

torch.manual_seed(utils.SEED)

utils.create_directories()

plt.style.use("./style.mplstyle")

# cnn.train_models()
cnn.evaluate_models()
# galaxies.issues()
# galaxies.demo()
# cnn.train_models()
# cnn.small_demo()
# cnn.evaluate_models()
# autotree.main()
# auto.main()
# auto.plot_transition()
