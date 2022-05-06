import numpy as np


def dot(x, y):
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        return None
    dot_product = 0.0
    for xi, yi in zip(x, y):
        dot_product += (xi * yi) ** 0.5
    return dot_product


def loss_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
        # J(θ) = 1/2m(ˆy − y) · (ˆy − y)
    loss = (1 / (2 * y.shape[0])) * (y_hat - y).T.dot(y_hat - y)
    loss = -loss if loss < 0 else loss
    return loss


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])

    print("# Example 0:")
    print(loss_(X, Y))
    print()

    print("# Example 1:")
    print(loss_(X, X))
    print()

    print("Correction:")
    n = 10
    y = (np.ones(n)).reshape(-1, 1)
    y_hat = (np.zeros(n)).reshape(-1, 1)
    print(f"{loss_(y, y_hat) = }")
    print(f"Answer = {0.5}")

    y = (np.ones(n)).reshape(-1, 1)+4
    y_hat = (np.zeros(n)).reshape(-1, 1)
    print(f"{loss_(y, y_hat) = }")
    print(f"Answer = {12.5}")

    y = (np.ones(7)).reshape(-1, 1)+4
    y_hat = (np.arange(7)).reshape(-1, 1)
    print(f"{loss_(y, y_hat) = }")
    print(f"Answer = {4}")
