import numpy as np


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


if __name__ == "__main__":
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])

    y = np.array([2, 14, -13, 5, 12, 4, -19])
    theta1 = np.array([3, 0.5, -6])
    print("# Example 0:")
    print(gradient(x, y, theta1))
    # Output:
    print("array([ -37.35714286, 183.14285714, -393. ])")
    print()

    print("# Example 1:")
    theta2 = np.array([0, 0, 0])
    print(gradient(x, y, theta2))
    # Output:
    print("array([ 0.85714286, 23.28571429, -26.42857143])")
    print()

    theta2 = np.array([[0], [0], [0], [0]])
    print(gradient(x, y, theta2))
