import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


class Minmax():
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


def model_load(poly):
    """In models.[csv/yml/pickle] one must find the parameters of all the
    models you have explored and trained. In space_avocado.py train the model based on
    the best hypothesis you find and load the other models from models.[csv/yml/pickle].
    Then evaluate and plot the different graphics as asked before.
    https://www.journaldev.com/15638/python-pickle-example
    '"""
    path = os.path.join(os.path.dirname(__file__), f"model_{poly}.pkl")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def multi_scatter(X, pred_price, true_price, costs):
    plot_dim = 2
    fig, axs_ = plt.subplots(plot_dim, plot_dim)
    axs = []
    for sublist in axs_:
        for item in sublist:
            axs.append(item)

    for idx_feature, feature in enumerate(X.T):
        for idx_pred, y_hat in enumerate(pred_price):
            c = ['r', 'y', 'm', 'b']
            color = c[idx_pred]
            axs[idx_feature].scatter(
                feature, y_hat, s=.1, c=color, label=f"Poly {idx_pred}")
        axs[idx_feature].scatter(
            feature, true_price, s=0.1, c='g', label="True")
        axs[idx_feature].legend()

    legend = [f"Pol {i}" for i in range(1, len(costs) + 1)]
    axs[-1].bar(legend, costs)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ms = ["v", "^", "<", ">"]
    c = ['r', 'y', 'm', 'b']

    ax.scatter(X[:, 1], X[:, 2], true_price, marker="o", c="g", label="True")
    for idx_pred, y_hat in enumerate(pred_price):
        ax.scatter(X[:, 1], X[:, 2], y_hat, marker=ms[idx_pred],
                   c=c[idx_pred], label=f"Poly {idx_pred}")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("../resources/space_avocado.csv")

    X = np.array(
        data[["weight", "prod_distance", "time_delivery"]]).reshape(-1, 3)
    Y = np.array(data["target"]).reshape(-1, 1)

    std_X = Minmax()
    std_X.fit(X)
    X_ = std_X.apply(X)

    loss = []
    prd = []
    for i in range(1, 5):
        print(f"{i}")
        X_poly = add_polynomial_features(X_, i)
        lr = model_load(i)
        y_hat = lr.predict(X_poly)
        cost = lr.cost_(Y, y_hat)
        print(f"{cost = }")
        loss.append(cost)
        prd.append(y_hat)

    multi_scatter(X, prd, Y, loss)
