import numpy as np


# Petit rappel : La régression linéaire est une régression polynomiale de degré 1.


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
    The matrix of polynomial features as a numpy.array, of shape m * n,
    containing the polynomial feature values for all training examples.
    None if x is an empty numpy.array.
    None if x or power is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if not isinstance(power, int):
        return None
    #  yˆ = θ0 + θ1*x + θ2x**2 + · · · + θnx**n
    res = x
    for i in range(2, power + 1):
        tmp = x ** (i)
        res = np.concatenate((res, tmp), axis=1)
    return res


x = np.arange(1, 6).reshape(-1, 1)
# Example 0:
print(add_polynomial_features(x, 3))


# Example 1:
print(add_polynomial_features(x, 6))


print("CORRECTION:")
print("\nTest 1:")
x1 = np.arange(1, 6).reshape(-1, 1)
x1_poly = add_polynomial_features(x1, 5)
print(f"{x1_poly = }")


print("\nTest 2:")
x2 = np.arange(10, 40, 10).reshape(-1, 1)
x2_poly = add_polynomial_features(x2, 5)
print(f"{x2_poly = }")


print("\nTest 3:")
x3 = np.arange(10, 40, 10).reshape(-1, 1)/10
x3_poly = add_polynomial_features(x3, 3)
print(f"{x3_poly = }")


# https://ledatascientist.com/regression-polynomiale-avec-python/#:~:text=La%20r%C3%A9gression%20polynomiale%20est%20une,est%20mod%C3%A9lis%C3%A9e%20comme%20un%20polyn%C3%B4me.
