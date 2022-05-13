import numpy as np

# scaling the data


def zscore(x: np.ndarray) -> np.ndarray:
    """Computes the normalized version of a non-empty numpy.array using the z-score standardization.
    Args:
    x: has to be an numpy.array, a vector.
    Return:
    x’ as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
    Raises:
    This function shouldn’t raise any Exception."""
    if not isinstance(x, np.ndarray):
        return None
    if len(x) == 0 or len(x.shape) != 2 or x.shape[1] != 1:
        return None
    try:
        mean = np.mean(x)
        # µ is the mean of x
    except (np.core._exceptions.UFuncTypeError, TypeError):
        return None
    std = np.std(x)
    # σ is the standard deviation of x
    res = np.zeros(x.shape)
    for i in range(x.size):
        # x0(i) = x(i)−µσ for i in 1, ..., m
        res[i] = (x[i] - mean) / std
    return res


X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
print("zscore X : \n\n", zscore(X), "\n")


Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
print("zscore Y : \n\n", zscore(Y))
