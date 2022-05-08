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


x = np.arange(1, 6).reshape(-1, 1)


theta1 = np.array([[5], [0]])
print("--- Example 1 ---")
print("")
print(predict_(x, theta1))
print("")

print("--- Example 2 ---")
print("")
theta2 = np.array([[0], [1]])
print(predict_(x, theta2))
print("")

print("--- Example 3 ---")
print("")
theta3 = np.array([[5], [3]])
print(predict_(x, theta3))
print("")

print("--- Example 4 ---")
print("")
theta4 = np.array([[-3], [1]])
print(predict_(x, theta4))
print("")


z = np.array([1, 2, 3, 4, 'a']).reshape(-1, 1)
theta = np.array([[5]])
print(predict_(z, theta))
