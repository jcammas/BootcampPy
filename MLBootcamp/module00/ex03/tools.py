import numpy as np


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    Returns:
    x as a numpy.array, a vector of shape m * 2.
    None if x is not a numpy.array.
    None if x is a empty numpy.array.
    Raises:
    This function should not raise any Exception"""
    if not isinstance(x, np.ndarray):
        return None
    try:
        shape = (x.shape[0], 1)
        ones = np.full(shape, 1)
        res = np.concatenate((ones, x), axis=1)
        return res
    except ValueError:
        return None


x = np.arange(1, 6).reshape((5, 1))
print("x before")
print("")
print(x)
print("")

print("x after")
print("")
print(add_intercept(x))
print("")

y = np.arange(1, 10).reshape((3, 3))
print("y before")
print("")
print(y)
print("")

print("y after")
print("")
print(add_intercept(y))
print("")

z = np.array([1, 2, 3, 4, 'a']).reshape(5, 1)
print('z before : \n', z, end='\n\n')
z = add_intercept(z)
print('z after : \n', z, end='\n\n')
