import numpy as np


# L2(θ)2 = Xnj=1θ2j


def iterative_l2(theta: np.ndarray) -> None:
    """Computes the L2 regularization of a non-empty numpy.array, with a for-loop.
    Args:
    theta: has to be a numpy.array, a vector of shape n’ * 1.
    Return:
    The L2 regularization as a float.
    None if theta in an empty numpy.array.
    None if theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    res = float(0)
    for i in range(1, theta.size):
        res = + theta[i][0] ** 2
    return res


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


def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two non-empty numpy.array,
    without any for loop. The two arrays must have the same shapes.
    Args:
    y: has to be an numpy.array, a vector of shape m * 1.
    y_hat: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a vector of shape n * 1.
    lambda_: has to be a float.
    Return:
    The regularized loss as a float.
    None if y, y_hat, or theta are empty numpy.array.
    None if y and y_hat do not share the same shapes.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if y.shape != y_hat.shape:
        return None
    # J(θ) = 1/2m[(ˆy − y) · (ˆy − y) + λ(θ0· θ0)]
    tmp = np.sum((y_hat - y) ** 2) + lambda_ * l2(theta)
    res = tmp / (2 * y.shape[0])
    return res


y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
y_hat = np.array([[3], [13], [-11.5], [5], [11], [5], [-20]])
theta = np.array([[1], [2.5], [1.5], [-0.9]])
# Example 1:
print(reg_loss_(y, y_hat, theta, .5))
# Output:
0.8503571428571429
# Example 2:
print(reg_loss_(y, y_hat, theta, .05))
# Output:
0.5511071428571429
# Example 3:
print(reg_loss_(y, y_hat, theta, .9))
# Output:
1.116357142857143

print("CORRECTION:")
y = np.arange(10, 100, 10)
y_hat = np.arange(9.5, 95, 9.5)
theta = np.array([-10, 3, 8])
lambda_ = 0.5
print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
print("Ans = 5.986111111111111")
print()

lambda_ = 5
print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
print("Ans = 24.23611111111111")
print()

y = np.arange(-15, 15, 0.1)
y_hat = np.arange(-30, 30, 0.2)
theta = np.array([42, 24, 12])
lambda_ = 0.5
print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
print("Ans = 38.10083333333307")
print()

lambda_ = 8
print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
print("Ans = 47.10083333333307")
print()
