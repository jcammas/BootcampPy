import numpy as np


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


x = np.arange(1, 13).reshape((4, -1))
theta1 = np.array([[5], [0], [0], [0]])
print(predict_(x, theta1))

theta2 = np.array([[0], [1], [0], [0]])
print(predict_(x, theta2))


theta3 = np.array([[-1.5], [0.6], [2.3], [1.98]])
print(predict_(x, theta3))

theta4 = np.array([[-3], [1], [2], [3.5]])
print(predict_(x, theta4))


print("CORRECTION:")
print("Test 1")
x = (np.arange(1, 13)).reshape(-1, 2)
theta = np.ones(3).reshape(-1, 1)
print(predict_(x, theta))
print("array([[ 4.], [ 8.], [12.], [16.], [20.], [24.]])")
print()

print("Test 2")
x = (np.arange(1, 13)).reshape(-1, 3)
theta = np.ones(4).reshape(-1, 1)
print(predict_(x, theta))
print("array([[ 7.], [16.], [25.], [34.]])")
print()

print("Test 3")
x = (np.arange(1, 13)).reshape(-1, 4)
theta = np.ones(5).reshape(-1, 1)
print(predict_(x, theta))
print("array([[11.], [27.], [43.]])")
print()
