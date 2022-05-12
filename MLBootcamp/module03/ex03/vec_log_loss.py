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


def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
    y: has to be an numpy.array, a vector of shape m * 1.
    y_hat: has to be an numpy.array, a vector of shape m * 1.
    eps: epsilon (default=1e-15)
    Return:
    The logistic loss value as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    # J(θ) = (−1 / m)[y · log(ˆy) + (~1 − y) · log(~1 − yˆ)]
    if y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    y_hat += eps
    return ((-1 / m) * (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))).sum()


if __name__ == "__main__":

    print("# Example 1:")
    y1 = np.array([[1]])
    x1 = np.array([[4]])
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    print(f"{vec_log_loss_(y1, y_hat1) = }")
    print("# Output:")
    print("0.01814992791780973")
    print()

    print("# Example 2:")
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    print(f"{vec_log_loss_(y2, y_hat2) = }")
    print("# Output:")
    print("2.4825011602474483")
    print()

    print("# Example 3:")
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    print(f"{vec_log_loss_(y3, y_hat3) = }")
    print("# Output:")
    print("2.9938533108607053")
    print()
