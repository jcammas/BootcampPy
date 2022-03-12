import numpy as np


def minmax(x):
    """"""
    x = x.reshape(-1, 1)  # array
    # x(i) - min(x) / max(x) - min(x)
    mandm = (x - x.min()) / (x.max() - x.min())
    return mandm


# Example 1:
X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
print("zscore X => ", minmax(X), "\n")

# Example 2:
Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
print("zscore Y => ", minmax(Y))
