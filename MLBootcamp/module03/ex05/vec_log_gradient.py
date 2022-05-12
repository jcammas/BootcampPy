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


def logistic_predict_(x, theta):
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
    x_ = add_intercept(x)
    y = sigmoid_(x_.dot(theta))
    return y


def vec_log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
    The gradient as a numpy.array, a vector of shape n * 1,
    containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if y, x or theta is not of the expected type.
    Raises:
    This function shoud not raise any Exception."""
    m = x.shape[0]
    y_hat = logistic_predict_(x, theta)
    x = add_intercept(x)
    # ∇(J) = 1/mX'T(hθ(X) − y)
    return x.T.dot(y_hat - y) / m


if __name__ == "__main__":

    print("# Example 1:")
    y1 = np.array([[1]])
    x1 = np.array([[4]])
    theta1 = np.array([[2], [0.5]])
    print(f"{vec_log_gradient(x1, y1, theta1) = }")

    print("# Example 2:")
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(f"{vec_log_gradient(x2, y2, theta2) = }")

    print("# Example 3:")
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(f"{vec_log_gradient(x3, y3, theta3) = }")

    print("CORRECTION:")
    print("Test:")
    x = np.array([[0, 0], [0, 0]])
    y = np.array([[0], [0]])
    theta = np.array([[0], [0], [0]])
    print(f"{vec_log_gradient(x, y, theta) = }")

    print("Test:")
    x = np.array([[1, 1], [1, 1]])
    y = np.array([[0], [0]])
    theta = np.array([[1], [0], [0]])
    print(f"{vec_log_gradient(x, y, theta) = }")

    print("Test:")
    x = np.array([[1, 1], [1, 1]])
    y = np.array([[1], [1]])
    theta = np.array([[1], [0], [0]])
    print(f"{vec_log_gradient(x, y, theta) = }")

    print("Test:")
    x = np.array([[1, 1], [1, 1]])
    y = np.array([[0], [0]])
    theta = np.array([[1], [1], [1]])
    print(f"{vec_log_gradient(x, y, theta) = }")
