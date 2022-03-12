import numpy as np


def dot(x, y):
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
    return dot(y_hat - y, y_hat - y)/y.size


# X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
# Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

# print(loss_(X, Y))
# print(loss_(X, X))
