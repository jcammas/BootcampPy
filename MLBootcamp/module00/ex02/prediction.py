import numpy as np


def simple_predict(X, theta):
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
    This function should not raise any Exception"""
    if not isinstance(X, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if theta.ndim != 2 or X.ndim != 2 or theta.shape[1] != 1 or X.shape[1] + 1 != theta.shape[0]:
        return None
    if theta.shape != (2, 1) or len(X) == 0 or len(theta) == 0 or (X.shape[1] != 1 and X.shape[0] <= 1) or\
            X.shape[1] != 1:
        return None
        # yˆ (i) = θ0 + θ1x(i) for i = 1, ..., m
    try:
        return np.array([theta[0] + (i * theta[1]) for i in X])
    except ValueError:
        return None


X = np.arange(1, 5).reshape(-1, 1)

print("theta1")
print("")
theta1 = np.array([[5], [0]])

print(simple_predict(X, theta1))
print("")
print("Do you understand why y_hat contains only 5’s here?")
print("(i * 0) + 5 = 5\n")

print("theta2")
print("")
theta2 = np.array([[0], [1]])
print(simple_predict(X, theta2))
print("")
print("Do you understand why y_hat == X here?")
print("(i * 1) + 0 = i\n")

print("theta3")
print("")

theta3 = np.array([[5], [3]])
print(simple_predict(X, theta3))
print("")
print("")

print("theta4")
print("")

theta4 = np.array([[-3], [1]])
print(simple_predict(X, theta4))

print("")


theta_invalid = np.array([[-3, 1]])
print("theta invalid: ", simple_predict(X, theta_invalid))
