import numpy as np
import src.FFNN, src.GradientDecent, src.utils

def compare_L1norm():


    x, y = utils.generate_regression_data(N=1000)
    x_train, x_test, y_train, y_test = utils.train_test_split(x, y)
    