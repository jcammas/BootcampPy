import numpy as np


# Understand and manipulate the concept of gradient descent in the case of multivariate
# linear regression. Implement a function to perform linear gradient descent (LGD) for
# multivariate linear regression.

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


def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of shape m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(x.shape) != 2 or x.shape[0] <= 1 or x.shape[1] <= 1 or\
            len(theta.shape) != 2 or theta.shape[1] != 1 or theta.shape[0] <= 1:
        return None
    try:
        t = add_intercept(x)
        y_hat = t.dot(theta)
    except (ValueError, np.core._exceptions.UFuncTypeError):
        return None
    return y_hat


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop.
        The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of dimensions n * 1, containg the result of
        the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    try:
        parenthesis = np.subtract(x.dot(theta), y)
        coef = x.dot(1/x.shape[0])
        # ∇(J) = 1/m * X'T * (X'θ − y)
        return np.transpose(coef).dot(parenthesis)
    except (ValueError, TypeError, np.core._exceptions.UFuncTypeError):
        return None


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
            examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
            examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient
            descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    x_ = add_intercept(x)
    for i in range(max_iter):
        dt = gradient(x_, y, theta).sum(axis=1)
        theta_update = (dt * alpha).reshape((-1, 1))
        theta = theta - theta_update
    return theta


x = np.array([[0.2, 2., 20.], [0.4, 4., 40.],
              [0.6, 6., 60.], [0.8, 8., 80.]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.], [1.], [1.], [1.]])
print("# Example 0:")
theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
print(theta2)


print("# Example 1:")
print(predict_(x, theta2))
