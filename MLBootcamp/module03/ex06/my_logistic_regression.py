# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

import numpy as np


class MyLogisticRegression():
    """Description: My personal logistic regression to classify things"""

    def __init__(self, theta: np.ndarray, alpha=0.001, max_iter=1000) -> None:
        if isinstance(theta, list):
            theta = np.asarray(theta).reshape(-1, 1)
        theta = theta.astype("float64")
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    @staticmethod
    def log_loss_(y, y_hat, eps=1e-15):
        """
        Computes the logistic loss value.
        Args:
        y: has to be an numpy.array, a vector of shape m * 1.
        y_hat: has to be an numpy.array, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
        Return:
        The logistic loss value as a float.
        None otherwise.
        Raises:
        This function should not raise any Exception.
        """
        if isinstance(y, (int, float)) == True:
            y = [float(y)]
        if isinstance(y_hat, (int, float)) == True:
            y_hat = [float(y_hat)]
        y = np.array(y)
        y_hat = np.array(y_hat)
        m = y.shape[0]
        return ((-1 / m) * (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))).sum()

    @staticmethod
    def add_intercept(x: np.ndarray) -> np.ndarray:
        """Adds a column of 1’s to the non-empty numpy.array x.
        Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        Returns:
        x as a numpy.array, a vector of shape m * 2.
        None if x is not a numpy.array.
        None if x is a empty numpy.array.
        Raises:
        This function should not raise any Exception"""
        if not isinstance(x, np.ndarray):
            return None
        try:
            shape = (x.shape[0], 1)
            ones = np.full(shape, 1)
            res = np.concatenate((ones, x), axis=1)
            return res
        except ValueError:
            return None

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

    def logistic_predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a vector of shape m * n.
        theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
        Return:
        y_hat: a numpy.array of shape m * 1, when x and theta numpy arrays
        with expected and compatible shapes.
        None: otherwise.
        Raises:
        This function should not raise any Exception.
        """
        # yˆ = sigmoid(X'·θ) = 1 / (1 + e^−X'.·0)
        x_ = self.add_intercept(x)
        y = self.sigmoid_(x_.dot(self.theta))
        return y

    def log_gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array, with a for-loop.
        The three arrays must have compatible shapes.
        Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
        Return:
        The gradient as a numpy.array, a vector of shapes n * 1,
        containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        m = x.shape[0]
        y_hat = self.logistic_predict_(x)
        x = self.add_intercept(x)
        # ∇(J) = 1/mX'T(hθ(X) − y)
        return x.T.dot(y_hat - y) / m

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
        theta = self.theta
        alpha = self.alpha
        if x.shape[0] != y.shape[0] or (x.shape[1] + 1) != theta.shape[0]:
            return None
        for _ in range(self.max_iter):
            new_theta = self.log_gradient(x, y)
            theta -= alpha * new_theta
        self.theta = theta
