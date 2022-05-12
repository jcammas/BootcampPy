from cgi import test
import pandas as pd
import numpy as np
import math


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    proportion: has to be a float, the proportion of the dataset that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible shapes.
    None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if x.shape[0] != y.shape[0]:
        return None
    ratio = int(x.shape[0] * proportion)
    shuffler = np.concatenate((x, y), 1)
    np.random.shuffle(shuffler)
    x, y = shuffler[..., :-1], shuffler[..., -1:]
    X_train, X_test, Y_train, Y_test = x[:
                                         ratio], x[ratio:], y[:ratio], y[ratio:]
    return X_train, X_test, Y_train, Y_test
