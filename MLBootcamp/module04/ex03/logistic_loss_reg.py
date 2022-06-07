import numpy as np


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.array, without any for-loop.
    Args:
    theta: has to be a numpy.array, a vector of shape n’ * 1.
    Return:
    The L2 regularization as a float.
    None if theta in an empty numpy.array.
    None if theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if theta.size == 0:
        return None
    res = theta[1:]
    return res.T.dot(res)


def reg_log_loss_(y, y_hat, theta, lambda_, eps=1e-15):
    """Computes the regularized loss of a logistic regression model from two non-empty numpy.array,
    without any for loop. The two arrays must have the same shapes.
    Args:
    y: has to be an numpy.array, a vector of shape m * 1.
    y_hat: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a vector of shape n * 1.
    lambda_: has to be a float.
    eps: has to be a float, epsilon (default=1e-15).
    Return:
    The regularized loss as a float.
    None if y, y_hat, or theta is empty numpy.array.
    None if y or y_hat have component ouside [0 ; 1]
    None if y and y_hat do not share the same shapes.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any exception."""
    # J(θ) = −1/m[y · log(ˆy) + (~1 − y) · log(~1 − yˆ)] + λ/2m(θ0· θ0)
    if y.shape != y_hat.shape:
        return None
    eps: float = 1e-15
    ones = np.ones(y.shape)
    m = y.shape[0]
    res = np.sum(y * np.log(y_hat + eps) + (ones - y)
                 * np.log(ones - y_hat + eps)) / -m
    res += (lambda_ * l2(theta)) / (2 * m)
    return res


y = np.array([[1], [1], [0], [0], [1], [1], [0]])
y_hat = np.array([[.9], [.79], [.12], [.04], [.89], [.93], [.01]])
theta = np.array([[1], [2.5], [1.5], [-0.9]])
# Example 1:
print(reg_log_loss_(y, y_hat, theta, .5))
# Output:
0.40824105118138265
# Example 2:
print(reg_log_loss_(y, y_hat, theta, .05))
# Output:
0.10899105118138264
# Example 3:
print(reg_log_loss_(y, y_hat, theta, .9))
# Output:
0.6742410511813826

print("CORRECTION:")
y = np.array([0, 1, 0, 1])
y_hat = np.array([0.4, 0.79, 0.82, 0.04])
theta = np.array([5, 1.2, -3.1, 1.2])

print(f"{reg_log_loss_(y, y_hat, theta, .5) = }")
print("Ans = 2.2006805525617885")
print()

print(f"{reg_log_loss_(y, y_hat, theta, .75) = }")
print("Ans = 2.5909930525617884")
print()

print(f"{reg_log_loss_(y, y_hat, theta, 1.0) = }")
print("Ans = 2.981305552561788")
print()

print(f"{reg_log_loss_(y, y_hat, theta, 0.0) = }")
print("Ans = 1.4200555525617884")
print()
