import numpy as np


class MyLinearRegression():

    def __init__(self, theta):
        self.theta = np.asarray(theta)

    def predict_(self, X):
        if self.theta.ndim != 2 or X.ndim != 2 or self.theta.shape[1] != 1 or X.shape[1] + 1 != self.theta.shape[0]:
            print("Incompatible dimension match between X and theta.")
            return None
        X = np.insert(X, 0, 1., axis=1)
        return X.dot(self.theta)

    def loss_elem_(self, X, Y):
        Y_hat = self.predict_(X)
        def loss_func(Y, Y_hat, m): return (Y - Y_hat) ** 2
        res = np.array([loss_func(i, j, len(Y)) for i, j in zip(Y, Y_hat)])
        return res

    def loss_(self, X, Y):
        Y_hat = self.predict_(X)
        if Y_hat is None:
            return None

        def m(X, Y): return ((Y_hat - Y)**2)/(2*X.shape[0])
        costs = m(X, Y)
        if costs is None:
            return None
        return costs.sum()

    def fit_(self, X, Y, alpha=0.001, n_cycle=10000):
        if self.theta.ndim != 2 or X.ndim != 2 or self.theta.shape[1] != 1 or X.shape[1] + 1 != self.theta.shape[0] or Y.shape[0] != X.shape[0]:
            print("Incompatible dimension match between X and theta.")
            return None

        m = X.shape[0]
        X = np.insert(X, 0, 1., axis=1)
        for i in range(n_cycle):
            hypothesis = X.dot(self.theta)
            parenthesis = np.subtract(hypothesis, Y)
            sigma = np.sum(np.dot(X.T, parenthesis), keepdims=True, axis=1)
            self.theta = self.theta - (alpha / m) * sigma
        return self.theta
