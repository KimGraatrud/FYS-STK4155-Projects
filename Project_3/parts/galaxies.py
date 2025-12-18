import numpy as np
from os import path
import matplotlib.pyplot as plt
from src.Dataset import GalaxyDataset
from src import utils


def page_through():
    # page through galaxies one at a time
    ds = GalaxyDataset(mode="train")

    idx = 0
    while True:
        img, z = ds[idx]

        n_channels = img.shape[0]

        fig, axs = plt.subplots(ncols=n_channels)
        for i, ax in enumerate(axs):
            ax.imshow(img[i])
            ax.set_axis_off()

        fig.suptitle(idx)

        plt.show()
        # fig.savefig(path.join(utils.FIGURES_URL, f"gal"))
        plt.close(fig)

        idx += 1


def demo():
    ds = GalaxyDataset(mode="train", normalize=False)

    idx = 99

    img, z = ds[idx]

    # band information taken from
    # https://en.wikipedia.org/wiki/Photometric_system
    bands = [
        ["g", 464, "Green"],
        ["r", 658, "Red"],
        ["i", 806, "Infrared"],
        ["z", 900, "Infrared"],
        ["y", 1020, "Infrared"],
    ]

    fig, axs = plt.subplots(ncols=5, figsize=(utils.APS_COL_W, 0.35 * utils.APS_COL_W))

    cmap = plt.colormaps["Greys"].reversed()
    for ax, c, info in zip(axs, img, bands):
        code, wl, name = info
        ax.imshow(c, cmap=cmap)
        # ax.imshow(np.log(c), cmap=cmap)
        ax.set_axis_off()
        ax.set_title(f"`{code}'\n{wl} nm\n{name}")

    fig.savefig(path.join(utils.FIGURES_URL, "gal_demo"))
    plt.close(fig)


def issues():
    ds = GalaxyDataset(mode="train")

    problem_idxs = [58, 229, 288, 294, 315]
    problem_channels = [0, 4, 0, 1, 0]

    fig, axs = plt.subplots(
        ncols=len(problem_idxs), figsize=(utils.APS_COL_W, 0.3 * utils.APS_COL_W)
    )

    letters = ["a", "b", "c", "d", "e"]
    cmap = plt.colormaps["Greys"].reversed()
    for ax, idx, chan, let in zip(axs, problem_idxs, problem_channels, letters):
        img = ds[idx][0][chan]
        ax.imshow(img, cmap=cmap)
        ax.set_axis_off()
        ax.set_title(f"{let}.")

    fig.savefig(path.join(utils.FIGURES_URL, "issues"))
    plt.close(fig)

    ds.close()
