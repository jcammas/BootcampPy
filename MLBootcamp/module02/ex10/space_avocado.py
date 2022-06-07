import pandas as pd
from my_linear_regression import MyLinearRegression
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


class utils():
    def __init__(self):
        self.min = 0.
        self.max = 0.

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def apply(self, X):
        res = (X - self.min) / (self.max - self.min + 1e-20)
        return res


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
    The matrix of polynomial features as a numpy.array, of shape m * n,
    containing the polynomial feature values for all training examples.
    None if x is an empty numpy.array.
    None if x or power is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if not isinstance(power, int):
        return None
    #  yˆ = θ0 + θ1*x + θ2x**2 + · · · + θnx**n
    res = x
    for i in range(2, power + 1):
        tmp = x ** (i)
        res = np.concatenate((res, tmp), axis=1)
    return res


def model_load():
    """In models.[csv/yml/pickle] one must find the parameters of all the
    models you have explored and trained. In space_avocado.py train the model based on
    the best hypothesis you find and load the other models from models.[csv/yml/pickle].
    Then evaluate and plot the different graphics as asked before.
    https://www.journaldev.com/15638/python-pickle-example
    '"""
    path = os.path.join(os.path.dirname(__file__), f"model_4.yml")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_model(X, predicted_price, avocado_price, loss):
    """"
    plot our model in order to materialize our data
    https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166"""

    ax = plt.figure().add_subplot(projection='3d')

    ax.scatter(X[:, 1], X[:, 2], avocado_price,
               marker="*", c="g", label="Avocado price")
    for i, y_hat in enumerate(predicted_price):
        ax.scatter(X[:, 1], X[:, 2], y_hat, marker=["o", "s", "+", "*"][i],
                   c=['r', 'y', 'm', 'b'][i], label=f"model {i}")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("../resources/space_avocado.csv")

    X = np.array(
        df[["weight", "prod_distance", "time_delivery"]]).reshape(-1, 3)
    Y = np.array(df["target"]).reshape(-1, 1)

    std_X = utils()
    std_X.fit(X)
    X_ = std_X.apply(X)

    costs = []
    preds = []
    X_model = add_polynomial_features(X_, 4)

    lr = model_load()
    lr.predict_(X_model)

    y_hat = lr.predict_(X_model)
    cost = lr.loss_(Y, y_hat)
    print(f"{cost = }")
    costs.append(cost)
    preds.append(y_hat)

    plot_model(X, preds, Y, costs)
