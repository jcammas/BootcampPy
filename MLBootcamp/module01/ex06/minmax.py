import numpy as np


def minmax(x):
    """Computes the normalized version of a non-empty numpy.array using the min-max standardization.
    Args:
    x: has to be an numpy.array, a vector.
    Return:
    x’ as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
    Raises:
    This function shouldn’t raise any Exception."""
    if not isinstance(x, np.ndarray) or len(x) == 0 or len(x.shape) != 2 or x.shape[1] != 1:
        return None
    x = x.reshape(-1, 1)  # array
    try:
        # x(i) - min(x) / max(x) - min(x)
        min_ = np.min(x)
        max_ = np.max(x)
        diff = max_ - min_
        res = np.zeros(x.shape)
        for i in range(x.size):
            res[i] = (x[i] - min_) / diff
    except (TypeError, np.core._exceptions.UFuncTypeError):
        return None
    return res


# Example 1:
X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
print("minamx X : \n\n", minmax(X), "\n")

# Example 2:
Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
print("minmax Y : \n\n", minmax(Y))

print('')
R = np.array([[]])
O = np.array([[0], [15], [-9], [7], [12], [3], ['a']])
R = np.array([[0], [15], [-9], [7], [12], [3], []], dtype=object)

print("should return None => ", minmax(R))
print("should return None => ", minmax(O))
print("should return None => ", minmax(R))
