import numpy as np
from tools import add_intercept as tool


def gradient(x, y, theta):
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
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    if x.shape[1] != 1 or y.shape[1] != 1 or theta.shape != (2, 1):
        return None
    try:
        x = tool(x)
        parenthesis = np.subtract(x.dot(theta), y)
        coef = x.dot(1/x.shape[0])
    except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
        return None
    return np.transpose(coef).dot(parenthesis)


x = np.array([[12.4956442], [21.5007972], [
             31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [
             45.7655287], [46.6793434], [59.5585554]])

theta1 = np.array([[2], [0.7]])
print(gradient(x, y, theta1))

theta2 = np.array([[1], [-0.4]])
print(gradient(x, y, theta2))
