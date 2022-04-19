import numpy as np


def simple_predict(x, theta):
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
    if theta.ndim != 2 or x.ndim != 2 or theta.shape[1] != 1 or x.shape[1] + 1 != theta.shape[0]:
        print("Error.")
        return None
    x = np.insert(x, 0, 1., axis=1)
    return x.dot(theta)


x = np.arange(1, 13).reshape((4, -1))
theta1 = np.array([[5], [0], [0], [0]])
print(simple_predict(x, theta1))


theta2 = np.array([[0], [1], [0], [0]])
print(simple_predict(x, theta2))


theta3 = np.array([[-1.5], [0.6], [2.3], [1.98]])
print(simple_predict(x, theta3))

theta4 = np.array([[-3], [1], [2], [3.5]])
print(simple_predict(x, theta4))
