import os
import matplotlib.pyplot as plt
from src import utils
from parts import B, F, weekly

plt.style.use("./style.mplstyle")

# create a home for figures
if not os.path.exists(utils.FIGURES_URL):
    os.mkdir(utils.FIGURES_URL)

# create a home for data
if not os.path.exists(utils.DATA_URL):
    os.mkdir(utils.DATA_URL)

if __name__ == "__main__":
    B.main()
    # F.main()
    # weekly.main()
