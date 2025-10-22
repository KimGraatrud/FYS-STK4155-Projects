import numpy as np


def mse_cost_der(targets, inputs):
    n = len(target)
    return 2/n * (inputs - targets)


def ridge_cost_der(targets, inputs, lmbda):
    n = len(target)
    return 2/n * (inputs - targets) + inputs*lmbda


def lasso_cost_der(targets, inputs, lmbda):
    n = len(target)
    return 2/n * (inputs - targets) + lmbda*np.sign(inputs)

