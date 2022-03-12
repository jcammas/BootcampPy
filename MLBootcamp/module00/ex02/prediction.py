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
    if theta.ndim != 2 or X.ndim != 2 or theta.shape[1] != 1 or X.shape[1] + 1 != theta.shape[0]:
        print("Error.")
        return None
        # yˆ (i) = θ0 + θ1x(i) for i = 1, ..., m
    return np.array([(i * theta[1]) + theta[0] for i in X])


X = np.arange(1, 6).reshape(-1, 1)


print("theta1")
print("")
theta1 = np.array([[5], [0]])
print(simple_predict(X, theta1))
print("")
print("Do you understand why y_hat contains only 5’s here?")
print("")

print("theta2")
print("")
theta2 = np.array([[0], [1]])
print(simple_predict(X, theta2))
print("")
print("Do you understand why y_hat == X here?")
print("")

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
