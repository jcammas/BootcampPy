import numpy as np
from matplotlib import pyplot as plt


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


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of shape m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if add_intercept(x).shape[1] != theta.shape[0]:
        return None
    i = add_intercept(x)
    if i.shape[1] != theta.shape[0]:
        return None
    return i.dot(theta)


def dot(x, y):
    if not isinstance(x, np.ndarray):
        return None
    if not isinstance(y, np.ndarray):
        return None
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        return None
    dot_product = 0.0
    for xi, yi in zip(x, y):
        dot_product += (xi * yi) / 2
    return dot_product


def loss_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    # if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape or y.shape[1] != 1 or y_hat.shape[1] != 1:
    #     return None
    try:
        return dot(y_hat - y, y_hat - y)/y.size
    except (TypeError, np.core._exceptions.UFuncTypeError):
        return None


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    plt.plot(x, y, 'o')

    plt.plot(x, theta[1] * x + theta[0])
    p = predict_(x, theta)
    loss = loss_(y, p)

    plt.title("Cost: %f" % loss)
    for x_i, p_i, y_i in zip(x, p, y):
        plt.plot([x_i, x_i], [y_i, p_i], 'r--')
    plt.show()


x = np.arange(1, 6).reshape(-1, 1)
y = np.array([[11.52434424], [10.62589482], [
             13.14755699], [18.60682298], [14.14329568]])


theta1 = np.array([[18], [-1]])
plot_with_loss(x, y, theta1)

theta2 = np.array([[14], [0]])
plot_with_loss(x, y, theta2)

theta3 = np.array([[12], [0.8]])
plot_with_loss(x, y, theta3)


plot_with_loss(np.array([0, 1]).reshape(2, 1), np.array(
    [0, 1]).reshape(2, 1), np.array([0, 1]).reshape(2, 1))
plot_with_loss(np.array([0, 1]).reshape(2, 1), np.array(
    [0, 1]).reshape(2, 1), np.array([1, 1]).reshape(2, 1))
plot_with_loss(np.array([0, 2]).reshape(2, 1), np.array(
    [0, 0]).reshape(2, 1), np.array([-1, 1]).reshape(2, 1))
