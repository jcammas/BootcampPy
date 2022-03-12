import numpy as np
from TinyStatistician import TinyStatistician as Ts


def zscore(x: np.ndarray) -> np.ndarray:
    """"""
    mean = Ts.mean(x)
    std = Ts.std(x)
    res = np.zeros(x.shape)
    for i in range(x.size):
        res[i] = (x[i] - mean) / std
    return res


X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
print("zscore X => ", zscore(X), "\n")


Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
print("zscore Y => ", zscore(Y))
