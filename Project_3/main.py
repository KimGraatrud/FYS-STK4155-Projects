import os
import matplotlib.pyplot as plt
from parts import cnn_training, cnn_plotting, galaxies, auto
from src import utils
from src.Dataset import GalaxyDataset
import torch

torch.manual_seed(utils.SEED)

utils.create_directories()

if not os.path.exists(utils.NORM_URL):
    utils.compute_normalization()

def produce_results():
    """
    Will produce most of the results seen in the report, given you have the dataset downloaded.
    Which is not recomended :D

    The only results that wont be produced are the ones which take many hours to generate.
    """

    galaxies.distribution()     # Plots the redshift sample distrobution
    galaxies.demo()             # Plots the example of the dataset
    galaxies.issues()           # Plots some of the 'problematic' galaxy image entries

    cnn_training.main()         # Trains the CNN models 
    cnn_plotting.main()         # Plots the CNN related results

    tree.main()                 # Trains and saves tree related models
    hybrid.main()               # Trains and saves hybrid model
    best_results.main()         # Prints / plots the best results from tree related and hybrid models





# Uncomment this if its causing matplotlib or LaTeX errors,
# alternatively comment out the latex lines in the file.
plt.style.use("./style.mplstyle")

print("Device:", utils.device)

print('Dataset lengths':)
for m in ["train", "test", "validate"]:
    ds = GalaxyDataset(mode=m)
    print(m, len(ds.z))
    ds.close()

# Produce the results.
produce_results()