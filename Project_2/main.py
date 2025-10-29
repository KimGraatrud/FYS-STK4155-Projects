import os
from src import utils
from parts import B, F

# create a home for figures
if not os.path.exists(utils.FIGURES_URL):
    os.mkdir(utils.FIGURES_URL)

# create a home for data
if not os.path.exists(utils.DATA_URL):
    os.mkdir(utils.DATA_URL)

# B.main()
F.main()
