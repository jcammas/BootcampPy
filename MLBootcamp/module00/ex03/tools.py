import numpy as np


def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    Returns:
    x as a numpy.array, a vector of shape m * 2.
    None if x is not a numpy.array.
    None if x is a empty numpy.array.
    Raises:
    This function should not raise any Exception"""
    try:
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        i = np.ones((x.shape[0], 1))
        return np.append(i, x, axis=1)
    except:
        return None


x = np.arange(1, 6).reshape((5, 1))
print("original array")
print("")
print(x)
print("")

print("add_intercept on original array")
print("")
print(add_intercept(x))


y = np.arange(1, 10).reshape((3, 3))
print("original array")
print("")
print(y)
print("")

print("add_intercept on original array")
print("")
print(add_intercept(y))