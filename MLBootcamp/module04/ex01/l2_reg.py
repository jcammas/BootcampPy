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


x = np.array([[2], [14], [-13], [5], [12], [4], [-19]]).reshape((-1, 1))
# Example 1:
print(iterative_l2(x))
# Output:
911.0
# Example 2:
print(l2(x))
# Output:
911.0
y = np.array([[3], [0.5], [-6]]).reshape((-1, 1))
# Example 3:
print(iterative_l2(y))
# Output:
36.25
# Example 4:
print(l2(y))
# Output:
36.25


print("CORRECTION:")

theta = np.ones(10)
print(f"{l2(theta) = }")
print("Ans = 9.0")
print()

theta = np.arange(1, 10)
print(f"{l2(theta) = }")
print("Ans = 284.0")
print()

theta = np.array([50, 45, 40, 35, 30, 25, 20, 15, 10,  5,  0])
print(f"{l2(theta) = }")
print("Ans = 7125.0")
print()
