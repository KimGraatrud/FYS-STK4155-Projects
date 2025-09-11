import numpy as np


def poly(x, d, intercept=False):
    """
    x: input
    d: degree
    """
    lowest = 0 if intercept else 1
    orders = np.arange(lowest, d + 1)
    exp, base = np.meshgrid(orders, x)
    return base**exp
