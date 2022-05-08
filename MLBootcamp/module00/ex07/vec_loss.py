import numpy as np


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


X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

print(loss_(X, Y))
print(loss_(X, X))


y3 = np.array([2, 14, -13, 5, 12, 4, -19])

print(loss_(y3, y3))


y_hat = np.array([1, 2, 3, 4])
y = np.array([0, 0, 0, 0])

print(loss_(y, y_hat))
