from os import path
import matplotlib.pyplot as plt
from src.Dataset import GalaxyDataset
from src import utils


def main():
    ds = GalaxyDataset(mode="test")

    img, z = ds[4]  # 4th galaxy is cool

    n_channels = img.shape[0]

    fig, axs = plt.subplots(ncols=n_channels)
    for i, ax in enumerate(axs):
        ax.imshow(img[i])
        ax.set_axis_off()

    fig.savefig(path.join(utils.FIGURES_URL, f"gal"))
    plt.close(fig)
