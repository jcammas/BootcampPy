import numpy as np


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


if __name__ == "__main__":

    x = np.array([[12.4956442], [21.5007972], [
        31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
        45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLinearRegression([2, 0.7])

    # Example 0.0: => ok
    print(lr1.predict_(x))
    print("")

    # # Example 0.1:
    print(lr1.loss_elem_(lr1.predict_(x), y))
    print("")

    # # Example 0.2:
    print(lr1.loss_(lr1.predict_(x), y))
    print("")

    # Example 1.0:
    lr2 = MyLinearRegression([1, 1], 5e-8, 1500000)
    lr2.fit_(x, y)
    print(lr2.theta)
    print("")

    # Example 1.1:
    print("# Example 1.1:")
    print(lr2.predict_(x))
    print("")

    # # Example 1.2:
    print(lr2.loss_elem_(lr2.predict_(x), y))
    print("")

    # Example 1.3:
    print(lr2.loss_(lr2.predict_(x), y))
    print("")
