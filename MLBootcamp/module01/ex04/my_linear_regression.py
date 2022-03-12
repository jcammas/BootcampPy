
import numpy as np


class MyLinearRegression():
    def __init__(self, theta: np.ndarray, alpha: float = 0.001, max_iter: int = 1000):
        if isinstance(theta, list):
            theta = np.asarray(theta).reshape(len(theta), 1)
        theta = theta.astype("float64")
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    @staticmethod
    def loss_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return (y_hat - y) ** 2 / (2 * y.shape[0]) * 10

    @staticmethod
    def loss_(y: np.ndarray, y_hat: np.ndarray) -> float:
        if MyLinearRegression.loss_elem_(y, y_hat) is None:
            return None
        return np.sum(MyLinearRegression.loss_elem_(y, y_hat)) / 10

    @staticmethod
    def add_intercept(x):
        if type(x) is not np.ndarray or len(x) == 0:
            return None
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        intercept = np.ones((x.shape[0], 1))
        return np.append(intercept, x, axis=1)

    def predict_(self, x):
        if self.add_intercept(x).shape[1] != self.theta.shape[0]:
            return None
        return np.dot(self.add_intercept(x), self.theta)

    def fit_(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.theta.ndim != 2 or x.ndim != 2 or self.theta.shape[1] != 1 or x.shape[1] + 1 != self.theta.shape[0] or y.shape[0] != x.shape[0]:
            return None
        m = x.shape[0]
        x = np.insert(x, 0, 1., axis=1)
        for i in range(self.max_iter):
            hypothesis = x.dot(self.theta)
            parenthesis = np.subtract(hypothesis, y)
            sigma = np.sum(np.dot(x.T, parenthesis), keepdims=True, axis=1)
            self.theta = self.theta - (self.alpha / m) * sigma
        return self.theta

    def mse_(self, y, y_hat):
        summed = 0.0
        for yi, yi_hat in zip(y, y_hat):
            summed += (yi - yi_hat) ** 2
        return summed/y.size
