import numpy as np


# The idea behind this exercice is to limit the possibility of having overfitting
# Here, we want to separate our data between to pool => train & test
# We also want to shuffle our test data in order to have something different every time

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    proportion: has to be a float, the proportion of the dataset that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible shapes.
    None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if x.shape[0] != y.shape[0]:
        return None
    ratio = int(x.shape[0] * proportion)
    shuffler = np.concatenate((x, y), 1)
    np.random.shuffle(shuffler)
    x, y = shuffler[..., :-1], shuffler[..., -1:]
    X_train, X_test, Y_train, Y_test = x[:
                                         ratio], x[ratio:], y[:ratio], y[ratio:]
    return X_train, X_test, Y_train, Y_test


x1 = np.array([[1], [42], [300], [10], [59]])
y = np.array([[0], [1], [0], [1], [0]])
# Example 0:
print()
print(data_spliter(x1, y, 0.8))
print()
# Example 1:
print()
print(data_spliter(x1, y, 0.5))
print()

x2 = np.array([[1, 42],
               [300, 10],
               [59, 1],
               [300, 59],
               [10, 42]])
y = np.array([[0], [1], [0], [1], [0]])
# Example 2:
print()
print(data_spliter(x2, y, 0.8))
print()

# Example 3:
print()
print(data_spliter(x2, y, 0.5))
print()


print("CORRECTION:\n")
x = np.ones(42).reshape((-1, 1
                         ))
y = np.ones(42).reshape((-1, 1))
ret = data_spliter(x, y, 0.42)
print(list(map(np.shape, ret)))


np.random.seed(42)
tmp = np.arange(0, 110).reshape(11, 10)
x = tmp[:, :-1]
y = tmp[:, -1].reshape((-1, 1))
ret = data_spliter(x, y, 0.42)
print(ret)

print(list(map(np.shape, ret)))
