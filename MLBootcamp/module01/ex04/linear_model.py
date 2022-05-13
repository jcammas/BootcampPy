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

    @staticmethod
    def mse_(y: np.ndarray, y_hat: np.ndarray) -> float:
        if y.shape != y_hat.shape:
            return None
        mse_elem = (y_hat - y) ** 2 / (y.shape[0])
        return np.sum(mse_elem)

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
            m = x.shape[0]
            x = self.add_intercept(x)
            res = x.T.dot(x.dot(self.theta) - y)
        except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
            return None
        return res / m

    def fit_(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.theta.ndim != 2 or x.ndim != 2 or self.theta.shape[1] != 1 or x.shape[1] + 1 != self.theta.shape[0] or y.shape[0] != x.shape[0]:
            return None
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(self.theta, np.ndarray) or not isinstance(self.alpha, float) or not isinstance(self.max_iter, int):
            return None
        if x.shape[1] != 1 or y.shape[1] != 1 or self.theta.shape != (2, 1):
            return None
        while self.max_iter > 0:
            # repeat until convergence: {
            #      compute ∇(J)
            #      θ0 := θ0 − α∇(J)0
            #      θ1 := θ1 − α∇(J)1
            #  }
            #  Where:
            #     • α (alpha) is the learning rate. It’s a small float number (usually between 0 and 1),
            #     • For now, "reapeat until convergence" will mean to simply repeat for max_iter (a
            #       number that you will choose wisely)
            new_theta = self.gradient(x, y)
            self.theta[0][0] -= self.alpha * new_theta[0][0]
            self.theta[1][0] -= self.alpha * new_theta[1][0]
            self.max_iter -= 1
        return self.theta


def draw_regression(x, y, MyLinearRegression):
    plt.plot(x, y, 'o', c='b')
    y_ = MyLinearRegression.predict_(x)
    plt.plot(x, y_, 'g--')
    plt.scatter(x, y_, c='g')
    plt.xlabel("Quantity of blue pills (in micrograms)")
    plt.ylabel("Space driving score")

    plt.show()


def plot_cost(x: np.ndarray, y: np.ndarray) -> None:
    plt.xlabel("$θ_1$")
    plt.ylabel("cost function $J(θ_0, θ_1)$")
    plt.grid()

    linear_model = MyLinearRegression(np.array([[0], [0]]), max_iter=500)
    thetas_0 = range(85, 95, 2)
    for t0 in thetas_0:
        linear_model.theta[0][0] = t0

        npoints = 100
        y_cost = [0] * npoints
        thetas1 = np.linspace(-15, -3.8, npoints)
        for i, t1 in enumerate(thetas1):
            linear_model.theta[1][0] = t1
            y_hat = linear_model.predict_(x)
            y_cost[i] = linear_model.loss_(y, y_hat)
        plt.plot(thetas1, y_cost, label="$J(θ_0=%d, θ_1)$" % t0)

    plt.legend()
    plt.show()


def plot2graphs(x: np.ndarray, y: np.ndarray) -> None:
    linear_model = MyLinearRegression(np.array([[89.0], [-8]]), max_iter=500)

    flag = 3
    if flag & 1:
        linear_model.fit_(x, y)
        y_hat = linear_model.predict_(x)
        draw_regression(x, y, y_hat)
    if flag & 2:
        plot_cost(x, y)


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
plot_cost(Xpill, Yscore)
