import numpy as np
import matplotlib.pyplot as plt
from src import ml, utils, regression
import os

plt.rcParams.update(
    {
        "figure.dpi": 600,
        "figure.constrained_layout.use": True,
    }
)

base = "./figures/"
p = lambda p: os.path.join(base, p)


# Part a
# def A():
def runge(x):
    return 1 / (1 + 25 * x**2)


# create data
n = 1e5
x = np.linspace(-1, 1, np.int32(n))
y_base = runge(x)
noise = 0.05 * np.random.normal(size=x.shape[0])
y = y_base + noise


# fit polynomial of given degree
def fit(x, y, degree):
    X = utils.poly_features(x, degree, intercept=True)

    theta = regression.OLS(X, y)

    y_pred = X @ theta

    return y_pred


# degrees = np.arange(1, 10)
# fracs = np.logspace(-3, 0, 5)
degrees = np.arange(1, 15)
fracs = np.logspace(-4, 0, 40)
intervals = np.int32(np.round(1 / fracs))  # use every ith datapoint

ds, ints = np.meshgrid(degrees, intervals)


def mse_and_r2(d, i):
    x_sampled = x[::i]
    y_sampled = y[::i]
    pred = fit(x_sampled, y_sampled, d)

    mse = utils.MSE(y_sampled, pred)
    r2 = utils.Rsqd(y_sampled, pred)

    return mse, r2


mse_and_r2_vec = np.vectorize(mse_and_r2, signature="(),()->(),()")

mse, r2 = mse_and_r2_vec(ds, ints)


# Plot it all
fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(8, 3))

# MSE
ax = axs[0]
cf = ax.pcolormesh(ds, n / ints, mse, shading="nearest")

ax.set_yscale("log")
ax.set_ylabel("Number of points")
ax.set_xlabel("Polynomial degree")
ax.set_title("MSE")

fig.colorbar(cf)

# R squared
ax = axs[1]
cf = ax.pcolormesh(ds, n / ints, r2, cmap="plasma", shading="nearest")
ax.set_title("$R^2$")
ax.set_xlabel("Polynomial degree")
fig.colorbar(cf)


fig.suptitle("OLS polynomial fitting")

fig.savefig(p("test"))


# A()

plt.close()
