import matplotlib.pyplot as plt
from src.parts import A, B, C, D, F, G, H

plt.rcParams.update(
    {
        # Figure
        "figure.dpi": 600,
        "figure.constrained_layout.use": True,
        # "savefig.format": "svg",
        # Plotting
        "lines.linewidth": 0.6,
        # Axes
        "axes.linewidth": 0.5,
        "axes.grid": "True",
        "grid.color": "black",
        "grid.alpha": 0.03,
        "axes.labelpad": 2.0,
        "axes.titlepad": 5.0,
        "ytick.major.width": 0.6,
        "xtick.major.width": 0.6,
        "ytick.minor.width": 0.3,
        "xtick.minor.width": 0.3,
        # Fonts
        "text.usetex": True,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
        "legend.fontsize": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # Legend
        "legend.handletextpad": 0.3,
        "legend.scatterpoints": 3,
        "legend.borderaxespad": 0.1,
        "legend.fancybox": False,
        "patch.linewidth": 0.5,
        "legend.columnspacing": 1,
        # "legend.edgecolor": "white",
    }
)


def runparts():
    A.main()
    print("A done")
    B.main()
    print("B done")
    C.main()
    print("C done")
    D.main()
    print("D done")
    G.main()
    print("G done")
    H.main()
    print("h done")


H.main()
# G.main()
