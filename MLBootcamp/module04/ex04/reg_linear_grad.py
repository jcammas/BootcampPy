import numpy as np


# class MyLinearRegression():
#     """"""

#     def __init__(self, thetas: np.ndarray, alpha: float = 0.001, max_iter: int = 1000):
#         if isinstance(thetas, list):
#             thetas = np.asarray(thetas).reshape(-1, 1)
#         thetas = thetas.astype("float64")
#         self.alpha = alpha
#         self.max_iter = max_iter
#         self.thetas = thetas

#     @staticmethod
#     def add_intercept(x: np.ndarray, axis: int = 1) -> np.ndarray:
#         """Adds a column of 1's to the non-empty numpy.ndarray x.
#         Args:
#             x: has to be an numpy.ndarray, a matrix of dimension m * n.
#         Returns:
#             X as a numpy.ndarray, a matrix of dimension m * (n + 1).
#             None if x is not a numpy.ndarray.
#             None if x is a empty numpy.ndarray.
#         Raises:
#             This function should not raise any Exception.
#         """
#         if not isinstance(x, np.ndarray) or x.size == 0:
#             return None
#         ones = np.ones((x.shape[0], 1))
#         res = np.concatenate((ones, x), axis=axis)
#         return res

#     @staticmethod
#     def loss_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
#         """
#         Description:
#             Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
#         Args:
#             y: has to be an numpy.ndarray, a vector.
#             y_hat: has to be an numpy.ndarray, a vector.
#         Returns:
#             J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
#             None if there is a dimension matching problem between X, Y or theta.
#         Raises:
#             This function should not raise any Exception.
#         """
#         if len(y) == 0 or len(y_hat) == 0:
#             return None
#         try:
#             def loss_func(y, y_, m): return (y - y_) ** 2
#             res = np.array([loss_func(i, j, len(y)) for i, j in zip(y, y_hat)])
#             return res
#         except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
#             return None

#     @staticmethod
#     def loss_(y: np.ndarray, y_hat: np.ndarray) -> float:
#         """Computes the half mean squared error of two non-empty numpy.ndarray,
#             without any for loop. The two arrays must have the same dimensions.
#         Args:
#             y: has to be an numpy.ndarray, a vector.
#             y_hat: has to be an numpy.ndarray, a vector.
#         Returns:
#             The half mean squared error of the two vectors as a float.
#             None if y or y_hat are empty numpy.ndarray.
#             None if y and y_hat does not share the same dimensions.
#         Raises:
#             This function should not raise any Exceptions.
#         """
#         if len(y) == 0:
#             return None
#         try:
#             res = abs((1 / (2 * y.shape[0])) *
#                       (y_hat - y).T.dot(y - y_hat).sum())
#             return res
#         except (np.core._exceptions.UFuncTypeError, TypeError):
#             return None

#     def predict_(self, x: np.ndarray) -> np.ndarray:
#         """Computes the prediction vector y_hat from two non-empty numpy.ndarray.
#         Args:
#             x: has to be an numpy.ndarray, a matrix of dimension m * n.
#             theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
#         Returns:
#             y_hat as a numpy.ndarray, a vector of dimension m * 1.
#             None if x or theta are empty numpy.ndarray.
#             None if x or theta dimensions are not appropriate.
#         Raises:
#             This function should not raise any Exception.
#         """
#         if not isinstance(x, np.ndarray):
#             return None
#         thetas = self.thetas
#         if (x.shape[1] + 1) != thetas.shape[0]:
#             return None
#         t = self.add_intercept(x)
#         y_hat = t.dot(thetas)
#         return y_hat

#     def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
#         """Computes a gradient vector from three non-empty numpy.ndarray,
#             without any for-loop. The three arrays must have the compatible dimensions.
#         Args:
#             x: has to be an numpy.ndarray, a matrix of dimension m * n.
#             y: has to be an numpy.ndarray, a vector of dimension m * 1.
#             theta: has to be an numpy.ndarray, a vector (n + 1) * 1.
#         Returns:
#             The gradient as a numpy.ndarray, a vector of dimensions n * 1,
#             containg the result of the formula for all j.
#             None if x, y, or theta are empty numpy.ndarray.
#             None if x, y and theta do not have compatible dimensions.
#         Raises:
#             This function should not raise any Exception.
#         """
#         tmp = (x.dot(self.thetas))
#         loss = tmp - y
#         res = (x.T.dot(loss)) / x.shape[0]
#         return res

#     def fit_(self, x: np.ndarray, y: np.ndarray) -> None:
#         """
#         Description:
#                 Fits the model to the training dataset contained in x and y.
#         Args:
#                 x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
#                         examples, 1).
#                 y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
#                         examples, 1).
#                 theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
#                 alpha: has to be a float, the learning rate
#                 max_iter: has to be an int, the number of iterations done during the gradient
#                         descent
#         Returns:
#                 new_theta: numpy.ndarray, a vector of dimension 2 * 1.
#                 None if there is a matching dimension problem.
#         Raises:
#                 This function should not raise any Exception.
#         """
#         if not isinstance(x, np.ndarray):
#             return None
#         if not isinstance(y, np.ndarray):
#             return None
#         x_ = self.add_intercept(x)
#         for _ in range(self.max_iter):
#             swp = self.gradient(x_, y).sum(axis=1)
#             tmp = (swp * self.alpha).reshape((-1, 1))
#             self.thetas = self.thetas - tmp
#         return self.thetas


# def reg_linear_grad(x: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_: float) -> np.ndarray:
#     if (0 in [x.size, y.size, theta.size] or x.shape[0] != y.shape[0] or
#             (x.shape[1] + 1) != theta.shape[0]):
#         return None
#     res = np.zeros(shape=(theta.shape))
#     m, n = x.shape
#     mylr = MyLinearRegression(theta)
#     y_hat = mylr.predict_(x)
#     for i in range(m):
#         y_diff = y_hat[i][0] - y[i][0]
#         res[0][0] += y_diff
#         for j in range(n):
#             res[j + 1][0] += (y_diff * x[i][j]) + \
#                 (lambda_ * theta[j + 1][0]) / m
#     res = res / m
#     return res


# def vec_reg_linear_grad(x: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_: float) -> np.ndarray:
#     """Computes the regularized linear gradient of three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible dimensions.
#     Args:
#       x: has to be a numpy.ndarray, a matrix of dimesion m * n.
#       y: has to be a numpy.ndarray, a vector of dimension m * 1.
#       theta: has to be a numpy.ndarray, a vector of dimension n * 1.
#       lambda_: has to be a float.
#     Returns:
#       A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all j.
#       None if y, x, or theta are empty numpy.ndarray.
#       None if y, x or theta does not share compatibles dimensions.
#     Raises:
#       This function should not raise any Exception.
#     """
#     if (0 in [x.size, y.size, theta.size] or x.shape[0] != y.shape[0] or
#             (x.shape[1] + 1) != theta.shape[0]):
#         return None
#     mylr = MyLinearRegression(theta)
#     m = x.shape[0]
#     y_hat = mylr.predict_(x)
#     x = mylr.add_intercept(x)
#     theta_prime = np.concatenate((np.array([[0]]), theta[1:, ...]), axis=0)
#     res = (x.T.dot(y_hat - y) + lambda_ * theta_prime) / m
#     return res


def add_intercept(x):
    vec_one = np.ones(x.shape[0])
    result = np.column_stack((vec_one, x))
    return result


def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray, with two
    ,→ for-loop. The three arrays must have compatible dimensions.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula
    ,→ for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles dimensions.
    Raises:
    This function should not raise any Exception.
    """
    return vec_reg_linear_grad(y, x, theta, lambda_)


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray, without
    ,→ any for-loop. The three arrays must have compatible dimensions.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula
    ,→ for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles dimensions.
    Raises:
    This function should not raise any Exception.
    """
    x_ = add_intercept(x)
    theta_ = theta.copy()
    theta_[0] = 0
    return ((x_.T.dot((x_.dot(theta) - y))) + lambda_ * theta_) / y.shape[0]


if __name__ == "__main__":
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])

    print("# Example 1.1:")
    res = reg_linear_grad(y, x, theta, 1)
    print(res)
    print("""
    # Output:
    array([[ -60.99 ],
    [-195.64714286],
    [ 863.46571429],
    [-644.52142857]])
    """)

    print("# Example 1.2:")
    res = vec_reg_linear_grad(y, x, theta, 1)
    print(res)
    print("""
    # Output:
    array([[ -60.99 ],
    [-195.64714286],
    [ 863.46571429],
    [-644.52142857]])
    """)

    print("# Example 2.1:")
    res = reg_linear_grad(y, x, theta, 0.5)
    print(res)
    print("""
    # Output:
    array([[ -60.99 ],
    [-195.86142857],
    [ 862.71571429],
    [-644.09285714]])
    """)

    print("# Example 2.2:")
    res = vec_reg_linear_grad(y, x, theta, 0.5)
    print(res)
    print("""
    # Output:
    array([[ -60.99 ],
    [-195.86142857],
    [ 862.71571429],
    [-644.09285714]])
    """)

    print("# Example 3.1:")
    res = reg_linear_grad(y, x, theta, 0.0)
    print(res)
    print("""
    # Output:
    array([[ -60.99 ],
    [-196.07571429],
    [ 861.96571429],
    [-643.66428571]])
    """)

    print("# Example 3.2:")
    res = vec_reg_linear_grad(y, x, theta, 0.0)
    print(res)
    print("""
    # Output:
    array([[ -60.99 ],
    [-196.07571429],
    [ 861.96571429],
    [-643.66428571]])
    """)

    print("CORRECTION:")
    x = np.arange(7, 49).reshape(7, 6)
    y = np.array([[1], [1], [2], [3], [5], [8], [13]])
    theta = np.array([[16], [8], [4], [2], [0], [0.5], [0.25]])

    print(f"{vec_reg_linear_grad(y, x, theta, 0.5) = }")
    print("""array([[  391.28571429],
            [11861.28571429],
            [12252.28571429],
            [12643.42857143],
            [13034.57142857],
            [13425.89285714],
            [13817.16071429]])""")
    print()

    print(f"{vec_reg_linear_grad(y, x, theta, 1.5) = }")
    print("""array([[  391.28571429],
            [11862.42857143],
            [12252.85714286],
            [12643.71428571],
            [13034.57142857],
            [13425.96428571],
            [13817.19642857]])""")
    print()

    print(f"{vec_reg_linear_grad(y, x, theta, 0.05) = }")
    print("""array([[  391.28571429],
            [11860.77142857],
            [12252.02857143],
            [12643.3       ],
            [13034.57142857],
            [13425.86071429],
            [13817.14464286]])""")
    print()
