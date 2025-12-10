import matplotlib.pyplot as plt
from parts import auto, autotree, cnn
from src import utils
import torch

torch.manual_seed(0)

utils.create_directories()

plt.style.use("./style.mplstyle")

# cnn.train_models()
# cnn.small_demo()
# cnn.evaluate_models()
# autotree.main()
auto.main()
# auto.plot_transition()
