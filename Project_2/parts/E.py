import numpy as np
import src.FFNN, src.GradientDecent, src.utils

def mseL1norm(predict, target, lam, nn):
    return (np.sum((target - predict) ** 2) / len(target)) + lam * np.abs()

def mseL1normder(predict, target, lam, W):

# def lamb_decorator(func, ...):
#     pass


def compare_L1norm():

    # Gen data
    x, y = utils.generate_regression_data(N=1000)
    x_train, x_test, y_train, y_test = utils.train_test_split(x, y)

    # Scale data
    x_tr_s, (train_mean, train_nz_std) = utils.scale_data(x_train)
    x_te_s = (x_test - train_mean) / train_nz_std


def compare_L2norm():

    # Gen data
    x, y = utils.generate_regression_data(N=1000)
    x_train, x_test, y_train, y_test = utils.train_test_split(x, y)

    # Scale data
    x_tr_s, (train_mean, train_nz_std) = utils.scale_data(x_train)
    x_te_s = (x_test - train_mean) / train_nz_std

