import numpy as np

#  Improve with the Gradient
#  From our multivariate linear hypothesis we can derive our multivariate gradient.
#  We can improve our model with gradient formula
#  Gradient Descent is an algorithm that finds the best-fit line for a given training dataset in a smaller number of iterations.


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
        # ∇(J) = 1 / mX'T * (X'θ − y)
        x = add_intercept(x)
        # we dot to deal with the vectorized stuf
        res = (x.T @ (x.dot(theta) - y))
        return res / x.shape[0]
    except (ValueError, TypeError, np.core._exceptions.UFuncTypeError):
        return None


if __name__ == "__main__":

    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta1 = np.array([[0], [3], [0.5], [-6]])
    print(gradient(x, y, theta1))

    theta2 = np.array([[0], [0], [0], [0]])
    print(gradient(x, y, theta2))

    x = np.ones(10).reshape(-1, 1)
    theta = np.array([[1], [1]])
    y = np.ones(10).reshape(-1, 1)
    print(gradient(x, y, theta))

    x = (np.arange(1, 13)).reshape(-1, 2)
    theta = np.array([[3], [2], [1]])
    y = np.arange(1, 13).reshape(-1, 1)
    print(gradient(x, y, theta))

    x = (np.arange(1, 13)).reshape(-1, 3)
    theta = np.array([[5], [4], [-2], [1]])
    y = np.arange(9, 13).reshape(-1, 1)
    print(gradient(x, y, theta))
