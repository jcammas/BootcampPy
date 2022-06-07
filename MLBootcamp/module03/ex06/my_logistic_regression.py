import numpy as np


class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """

    def __init__(self, theta: np.ndarray, alpha: float = 1e-3, max_iter: int = 1000):
        if isinstance(theta, list):
            theta = np.asarray(theta).reshape(-1, 1)
        theta = theta.astype("float64")
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    @staticmethod
    def loss_(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15) -> float:
        """
        Computes the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
            y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
            eps: has to be a float, epsilon (default=1e-15)
        Returns:
            The logistic loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        if y.shape != y_hat.shape:
            return None
        ones = np.ones(y.shape)
        m = y.shape[0]
        res = np.sum(y * np.log(y_hat + eps) + (ones - y)
                     * np.log(ones - y_hat + eps)) / -m
        return res

    @staticmethod
    def sigmoid_(x: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid of a vector.
        Args:
        x: has to be an numpy.array, a vector
        Return:
        The sigmoid value as a numpy.array.
        None otherwise.
        Raises:
        This function should not raise any Exception.
        """
        # sigmoid(x) = 1 / (1 + e^−x)
        if x.size == 0:
            return None
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def add_intercept(x: np.ndarray, axis: int = 1) -> np.ndarray:
        """Adds a column of 1's to the non-empty numpy.ndarray x.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
        Returns:
            X as a numpy.ndarray, a matrix of dimension m * (n + 1).
            None if x is not a numpy.ndarray.
            None if x is a empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        ones = np.ones((x.shape[0], 1))
        res = np.concatenate((ones, x), axis=axis)
        return res

    @staticmethod
    def sigmoid_(x: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid of a vector.
        Args:
            x: has to be an numpy.ndarray, a vector
        Returns:
            The sigmoid value as a numpy.ndarray.
            None if x is an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        if x.size == 0:
            return None
        return 1 / (1 + np.exp(-x))

    def predict_(self, x: np.ndarray) -> np.ndarray:
        """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exception.
        """
        # yˆ = sigmoid(X'·θ) = 1 / (1 + e^−X'.·0)
        x_ = self.add_intercept(x)
        y = self.sigmoid_(x_.dot(self.theta))
        return y

    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes a gradient vector from three non-empty numpy.ndarray, without any a for-loop.
            The three arrays must have compatible dimensions.
        Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector (n + 1) * 1.
        Returns:
        The gradient as a numpy.ndarray, a vector of dimensions (n + 1) * 1, containing
            the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """
        theta = self.theta
        if (0 in [x.size, y.size, theta.size] or x.shape[0] != y.shape[0] or
                (x.shape[1] + 1) != theta.shape[0]):
            return None
        y_hat = self.predict_(x)
        x = self.add_intercept(x)
        res = x.T.dot(y_hat - y) / x.shape[0]
        return res

    def fit_(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: a matrix of dimension m * n: (number of training examples, number of features).
            y: a vector of dimension m * 1: (number of training examples, 1).
            theta: a vector of dimension (n + 1) * 1: (number of features + 1, 1).
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
            new_theta: numpy.ndarray, a vector of dimension (number of features + 1, 1).
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        for _ in range(self.max_iter):
            grad = self.gradient_(x, y).sum(axis=1)
            self.theta = self.theta - (self.alpha * grad).reshape((-1, 1))
        return self
