import os
import multiprocessing
import matplotlib.pyplot as plt
from src import utils
from parts import (
    heatmap,
    overfit,
    relu_comp,
    batches,
    eta,
    actvns,
    regularization,
    confusion,
    descent,
)

plt.style.use("./style.mplstyle")

# create a home for figures
if not os.path.exists(utils.FIGURES_URL):
    os.mkdir(utils.FIGURES_URL)

# create a home for data
if not os.path.exists(utils.DATA_URL):
    os.mkdir(utils.DATA_URL)

if __name__ == "__main__":
    print("multiprocessing.cpu_count()", multiprocessing.cpu_count())

    print("====== heatmap =======")
    heatmap.main()
    print("====== overfit =======")
    overfit.main()
    print("====== relu_comp =======")
    relu_comp.main()
    print("====== batches =======")
    batches.main()
    print("====== eta =======")
    eta.main()
    print("===== actvns ======")
    actvns.main()
    print("===== regularization ======")
    regularization.main()
    print("===== confusion ======")
    confusion.main()
    print("===== descent ======")
    descent.main()
