import matplotlib.pyplot as plt
from parts import auto, autotree
from src import utils
import torch

torch.manual_seed(0)

utils.create_directories()

plt.style.use("./style.mplstyle")

autotree.main()
# auto.main()
# auto.plot_transition()
# channels.main()
