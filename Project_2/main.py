import os
from src import utils
from parts import B

# create a home for figures
if not os.path.exists(utils.FIGURES_URL):
    os.mkdir(utils.FIGURES_URL)

B.main()
