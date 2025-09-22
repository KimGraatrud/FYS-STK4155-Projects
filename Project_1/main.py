import matplotlib.pyplot as plt
from src.parts import A, B, C

plt.rcParams.update(
    {
        "figure.dpi": 600,
        "figure.constrained_layout.use": True,
        "legend.scatterpoints": 4,
        # "savefig.format": "svg",
    }
)

# A.main()
# B.main()
# C.main()
C.eta_n_relationship()
