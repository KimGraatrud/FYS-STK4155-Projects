import os
import numpy as np
import matplotlib.pyplot as plt
from src import utils

plt.style.use("./style.mplstyle")

# arb. data
x = np.linspace(0, 10, 200)
y1 = x
y2 = x**2
y3 = np.exp(x)
y4 = np.pow(x, x)

# plotting
fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W))

ax.plot(x, y1, label="$x$", c="tan")
ax.plot(x, y2, label="$x^2$", c="cadetblue")
ax.plot(x, y3, label="$e^x$", c="thistle")
ax.plot(x, y4, label="$x^x$", c="indianred")

ax.axhline(1, c="k", ls="--", lw=0.5)

ax.set_yscale("log")

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.set_title("Function growth")

ax.legend()

fig.savefig(os.path.join(utils.FIGURES_URL, "example"))

plt.close(fig)
