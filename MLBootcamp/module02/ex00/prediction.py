import numpy as np


def simple_predict(x, theta):
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
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(x.shape) != 2 or x.shape[0] <= 1 or x.shape[1] <= 1 or\
            len(theta.shape) != 2 or theta.shape[1] != 1 or theta.shape[0] <= 1:
        return None
    try:
        m, n = x.shape
        y_hat = np.zeros((m, 1))
        # vector of dimension m / yˆ(i) is the ith component of the yˆ vector: the predicted value for the ith example
        y_hat[..., 0] = theta[0, 0]
        for i in range(m):
            for j in range(n):
                # the idea here is to set up a beginning of multivariate hypothesis prediction
                # for that, we need more than a vector, we need a matrix
                # that's why we have a matrix of dimension (m x n) where n is the number of features that characterize each training example => design matrix
                # we want our hypothesis to take more than one feature into account with this formula
                # yˆ (i) = θ0 + θ1x(i) for i = 1, ..., m
                y_hat[i][0] += x[i][j] * theta[j + 1][0]
    except ValueError:
        return None
    return y_hat


X = np.arange(1, 13).reshape(4, -1)
theta1 = np.array([[5], [0], [0], [0]])
print(simple_predict(X, theta1))

theta2 = np.array([[0], [1], [0], [0]])
print(simple_predict(X, theta2))

theta3 = np.array([[-1.5], [0.6], [2.3], [1.98]])
print(simple_predict(X, theta3))

theta4 = np.array([[-3], [1], [2], [3.5]])
print(simple_predict(X, theta4))

print("CORRECTION:")
print("Test 1")
x = (np.arange(1, 13)).reshape(-1, 2)
theta = np.ones(3).reshape(-1, 1)
print(simple_predict(x, theta))


print("Test 2")
x = (np.arange(1, 13)).reshape(-1, 3)
theta = np.ones(4).reshape(-1, 1)
print(simple_predict(x, theta))


print("Test 3")
x = (np.arange(1, 13)).reshape(-1, 4)
theta = np.ones(5).reshape(-1, 1)
print(simple_predict(x, theta))
