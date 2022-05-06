import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class MyLinearRegression():
    def __init__(self, theta: np.ndarray, alpha: float = 0.001, max_iter: int = 1000):
        if isinstance(theta, list):
            theta = np.asarray(theta).reshape(len(theta), 1)
        theta = theta.astype("float64")
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        if len(y) == 0 or len(y_hat) == 0:
            return None
        try:
            return (y_hat - y) ** 2 / (2 * y.shape[0]) * 10
        except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
            return None

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        if len(y) == 0:
            return None
        if self.loss_elem_(y, y_hat) is None:
            return None
        try:
            return np.sum(self.loss_elem_(y, y_hat)) / 10
        except (np.core._exceptions.UFuncTypeError, TypeError):
            return None

    def mse_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape or y.shape[1] != 1 or y_hat.shape[1] != 1:
            return None
        summed = 0.0
        for yi, yi_hat in zip(y, y_hat):
            summed += (yi - yi_hat) ** 2
        return summed/y.size

    def add_intercept(self, x):
        if type(x) is not np.ndarray or len(x) == 0:
            return None
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        intercept = np.ones((x.shape[0], 1))
        return np.append(intercept, x, axis=1)

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray):
            return None
        if self.add_intercept(x).shape[1] != self.theta.shape[0]:
            return None
        return np.dot(self.add_intercept(x), self.theta)

    def gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array, without any for loop.
        The three arrays must have compatible shapes.
        Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
        Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta is an empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
        Raises:
        This function should not raise any Exception."""
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(self.theta, np.ndarray):
            return None
        if len(x) == 0 or len(y) == 0 or len(self.theta) == 0:
            return None
        if x.shape[1] != 1 or y.shape[1] != 1 or self.theta.shape != (2, 1):
            return None
        try:
            x = self.add_intercept(x)
            parenthesis = np.subtract(x.dot(self.theta), y)
            coef = x.dot(1/x.shape[0])
        except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
            return None
        return np.transpose(coef).dot(parenthesis)

    def fit_(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.theta.ndim != 2 or x.ndim != 2 or self.theta.shape[1] != 1 or x.shape[1] + 1 != self.theta.shape[0] or y.shape[0] != x.shape[0]:
            return None
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(self.theta, np.ndarray) or not isinstance(self.alpha, float) or not isinstance(self.max_iter, int):
            return None
        if x.shape[1] != 1 or y.shape[1] != 1 or self.theta.shape != (2, 1):
            return None
        for i in range(self.max_iter):
            g = self.gradient(x, y)
            self.theta = self.theta - (self.alpha * g)


def draw_regression(x, y, MyLinearRegression):
    plt.plot(x, y, 'o', c='b')
    y_ = MyLinearRegression.predict_(x)
    plt.plot(x, y_, 'g--')
    plt.scatter(x, y_, c='g')
    plt.xlabel("Quantity of blue pills (in micrograms)")
    plt.ylabel("Space driving score")

    plt.show()


def draw_cost_function(x, y):
    plt.ylim((10, 50))
    plt.xlim((-14, -4))
    ran = 15
    upd = ran * 2 / 6
    for t0 in np.arange(89 - ran, 89 + ran, upd):
        cost_list = []
        theta_list = []
        for t1 in np.arange(-80 - 100, -8 + 100, 0.1):
            lr = MyLinearRegression(theta=[t0, t1], alpha=1e-3, max_iter=50000)
            y_ = lr.predict_(x)
            mse_c = lr.loss_(y, y_)
            cost_list.append(mse_c)
            theta_list.append(t1)
        label = "θ[0]=" + str(int(t0 * 10) / 10)
        plt.plot(theta_list, cost_list, label=label)
    plt.xlabel("Theta1")
    plt.ylabel("Cost function J(Theta0, Theta1)")
    plt.show()
    plt.cla()


data = pd.read_csv("../resources/are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
Yscore = np.array(data["Score"]).reshape(-1, 1)
# Example 1:
linear_model1 = MyLinearRegression(np.array([[89.0], [-8]]))
Y_model1 = linear_model1.predict_(Xpill)
print("MyLinearRegression =>", linear_model1.mse_(Yscore, Y_model1))
print("sklearn =>", mean_squared_error(Yscore, Y_model1))

# Example 2:
linear_model2 = MyLinearRegression(np.array([[89.0], [-6]]))
Y_model2 = linear_model2.predict_(Xpill)

print("MyLinearRegression =>", linear_model2.mse_(Yscore, Y_model2))

print("sklearn =>", mean_squared_error(Yscore, Y_model2))


draw_regression(Xpill, Yscore, linear_model1)
draw_cost_function(Xpill, Yscore)
