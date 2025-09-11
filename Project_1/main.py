import numpy as np
import matplotlib.pyplot as plt
from src import ml, features
import os


# Part a

# Part b

# Part c

base = "./figures/"
p = lambda p: os.path.join(base, p)

x = np.linspace(-1, 1, 100)
# y = np.sin(x) ** 3
y = np.sin(x) ** 3 + 0.1 * np.random.normal(size=100)

X = features.poly(x, 3, intercept=True)

gd = ml.GD(
    n_iterations=1e5,
    mass=0,
)
t = gd.ADAM(X, y)

out = X @ t

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, out)
fig.savefig(p("test"))


# descent.ols()
