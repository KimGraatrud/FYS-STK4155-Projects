import matplotlib.pyplot as plt
from src.parts import A, B, C, D

plt.rcParams.update(
    {
        "figure.dpi": 600,
        "figure.constrained_layout.use": True,
        "legend.scatterpoints": 4,
        "axes.grid": "True",
        "grid.color": "black",
        "grid.alpha": 0.03,
        # "savefig.format": "svg",
    }
)

# A.main()
# B.main()
# C.main()
D.main()
