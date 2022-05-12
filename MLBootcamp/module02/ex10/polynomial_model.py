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
