from xml.dom.expatbuilder import theDOMImplementation
import numpy as np

# Create a function that takes a matrix X of dimensions (m x n) and an integer p as input,
# and returns a matrix of dimension (m x (np)). For each column xj of the matrix X, the
# new matrix contains xj raised to the power of k, for k = 1, 2, ..., p


def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range
    of 1 up to the power given in argument.
    Args:
    x: has to be an numpy.array, where x.shape = (m,n) i.e. a matrix of shape m * n.
    power: has to be a positive integer, the power up to which the columns of matrix x
    are going to be raised.
    Return:
    - The matrix of polynomial features as a numpy.array, of shape m * (np),
    containg the polynomial feature values for all training examples.
    - None if x is an empty numpy.array.
    - None if x or power is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    res = x
    for i in range(2, power - 1):
        raised = x ** i
        res = np.concatenate((res, raised), axis=1)
    return res


x = np.arange(1, 11).reshape(5, 2)
# Example 1:
print(add_polynomial_features(x, 3))

# Example 2:
print(add_polynomial_features(x, 5))

print("CORRECTION:")
x1 = np.ones(10).reshape(5, 2)
print(f"{add_polynomial_features(x1, 3) = }")
print("""[[   1    1    1    1    1    1]
        [   1    1    1    1    1    1]
        [   1    1    1    1    1    1]
        [   1    1    1    1    1    1]
        [   1    1    1    1    1    1]]""")

x = np.arange(1, 6, 1).reshape(-1, 1)
X = np.hstack((x, -x))
print(f"{add_polynomial_features(X, 3) = }")
print("""array([[   1,   -1,    1,    1,    1,   -1],
            [   2,   -2,    4,    4,    8,   -8],
            [   3,   -3,    9,    9,   27,  -27],
            [   4,   -4,   16,   16,   64,  -64],
            [   5,   -5,   25,   25,  125, -125]])""")
