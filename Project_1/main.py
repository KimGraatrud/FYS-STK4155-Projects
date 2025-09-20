import matplotlib.pyplot as plt
from src.parts import A

plt.rcParams.update(
    {
        "figure.dpi": 600,
        "figure.constrained_layout.use": True,
        "legend.scatterpoints": 4,
    }
)

A.main()
