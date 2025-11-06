import os
import matplotlib.pyplot as plt
from src import utils
from parts import heatmap, overfit, relu_comp, test, batches

plt.style.use("./style.mplstyle")

# create a home for figures
if not os.path.exists(utils.FIGURES_URL):
    os.mkdir(utils.FIGURES_URL)

# create a home for data
if not os.path.exists(utils.DATA_URL):
    os.mkdir(utils.DATA_URL)

if __name__ == "__main__":
    # print("====== heatmap =======")
    # heatmap.main()
    # print("====== overfit =======")
    # overfit.main()
    # print("====== relu_comp =======")
    # relu_comp.main()

    batches.main()

    # test.main()
    # F.main()
    # weekly.main()
