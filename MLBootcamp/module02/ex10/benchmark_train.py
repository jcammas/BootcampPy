from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from my_linear_regression import MyLinearRegression as MyLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
path = os.path.join(os.path.dirname(__file__), '..', 'ex07')
sys.path.insert(1, path)
# path = os.path.join(os.path.dirname(__file__), '..', 'ex08')
# sys.path.insert(1, path)
# from polynomial_train import continuous_plot
path = os.path.join(os.path.dirname(__file__), '..', 'ex09')
sys.path.insert(1, path)


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


def model_save(data, poly):
    path = os.path.join(os.path.dirname(__file__), f"model_{poly}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def model_load(poly):
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


def one_loop(X, Y, poly=1):
    print(f"Poly {poly}")
    X_poly = add_polynomial_features(X, poly)
    X_train, X_test, Y_train, Y_test = data_spliter(X_poly, Y, 0.8)

    theta = [0] * (poly * X.shape[1] + 1)
    alpha = 1e-2

    lr = MyLR(thetas=theta, alpha=alpha, max_iter=100000)

    lr.fit_(X_train, Y_train)
    model_save(lr, poly)
    cost = lr.cost_(Y_test, lr.predict(X_test))
    print(f"{cost = }")
    return cost, lr, lr.predict(X_poly)


if __name__ == "__main__":
    data = pd.read_csv("../resources/space_avocado.csv")

    X = np.array(
        data[["weight", "prod_distance", "time_delivery"]]).reshape(-1, 3)
    Y = np.array(data["target"]).reshape(-1, 1)

    std_X = Minmax()
    std_X.fit(X)
    X_ = std_X.apply(X)

    costs = []
    preds = []
    for i in range(1, 5):
        c, lr, pred = one_loop(X_, Y, poly=i)
        costs.append(c)
        preds.append(pred)

    multi_scatter(X, preds, Y, costs)
