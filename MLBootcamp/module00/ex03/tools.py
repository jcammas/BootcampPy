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
    if not isinstance(x, np.ndarray):
        return None
    if len(x.shape) != 2 or x.shape[1] != 1:
        return None
    try:
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        i = np.ones((x.shape[0], 1))
        return np.append(i, x, axis=1)
    except ValueError:
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


x = np.arange(1, 6).reshape((5, 1))
print('x before : \n', x, end='\n\n')
x = add_intercept(x)
print('x after : \n', x, end='\n\n')

y = np.arange(1, 10).reshape((3, 3))
print('y before : \n', y, end='\n\n')
y = add_intercept(y)
print('y after : \n', y, end='\n\n')

z = np.array([1, 2, 3, 4, 'a']).reshape(5, 1)
print('z before : \n', z, end='\n\n')
z = add_intercept(z)
print('z after : \n', z, end='\n\n')
