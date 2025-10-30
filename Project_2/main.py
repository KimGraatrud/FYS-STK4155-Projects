import os
import matplotlib.pyplot as plt
from src import utils
from parts import B, F

plt.rcParams.update(
    {
        # Figure
        "figure.dpi": 600,
        "figure.constrained_layout.use": True,
    }
)

# create a home for figures
if not os.path.exists(utils.FIGURES_URL):
    os.mkdir(utils.FIGURES_URL)

# create a home for data
if not os.path.exists(utils.DATA_URL):
    os.mkdir(utils.DATA_URL)

# B.main()
F.main()
