import matplotlib.pyplot as plt
from src.parts import A, B, D, H, G

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
        "text.usetex": False,       # Got some wierd latex error when i had this enabled...
        # "mathtext.fontset": "cm",
        # "font.family": "serif",
        # "font.serif": "Computer Modern Roman",
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


A.main()
print('A')
B.main()
print('b')

# C.main()
# print('c')


D.main()
print('d')

G.main()
print('g')

# H.main()