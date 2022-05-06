# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# import sys
# warnings.filterwarnings('ignore')

# ALLOWED_TYPES = [int, float, np.int64, np.float64]


# class MyLinearRegression():

#     def __init__(self, theta, alpha=0.001, n_cycle=2000):
#         if not MyLinearRegression.verif_params(theta):
#             sys.exit('Invalid theta param')
#         self.alpha = alpha
#         self.n_cycle = n_cycle
#         self.theta = np.asarray(theta).reshape(-1, 1)

#     @staticmethod
#     def verif_params(*args):
#         for arg in args:
#             if not isinstance(arg, list) and not isinstance(arg, np.ndarray):
#                 return False
#             for val in arg:
#                 if type(val) not in ALLOWED_TYPES:
#                     if isinstance(val, np.ndarray) or isinstance(val, list):
#                         try:
#                             tmp = val.sort()
#                             for v in val:
#                                 if type(v) not in ALLOWED_TYPES:
#                                     return False
#                         except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
#                             return False
#                         continue
#                     else:
#                         return False
#         return True

#     def predict_(self, X):
#         if self.theta.ndim != 2 or X.ndim != 2 or self.theta.shape[1] != 1 or X.shape[1] + 1 != self.theta.shape[0]:
#             print("Incompatible dimension match between X and theta.")
#             return None
#         if not MyLinearRegression.verif_params(X):
#             return None
#         X = np.insert(X, 0, 1., axis=1)
#         return X.dot(self.theta)

#     def loss_elem_(self, X, Y):
#         if not MyLinearRegression.verif_params(X, Y):
#             return None
#         Y_hat = self.predict_(X)
#         res = np.array([(Y - Y_hat) ** 2])
#         return res

#     def loss_(self, X, Y):
#         if not MyLinearRegression.verif_params(X, Y):
#             return None
#         Y_hat = self.predict_(X)
#         if Y_hat is None:
#             return None
#         return np.sum((Y_hat - Y)**2)/(2*X.shape[0])

#     @staticmethod
#     def add_intercept(x):
#         """
#         Adds a column of 1's to the non-empty numpy.array x
#         """
#         try:
#             new_array = np.array(x, dtype=float)
#         except (np.core._exceptions.UFuncTypeError, ValueError):
#             return None
#         intercept_ = np.ones((1, len(x)), dtype=float)
#         return np.insert(new_array, 0, intercept_, axis=1)

#     def fit_(self, x, y):
#         if not MyLinearRegression.verif_params(x, y):
#             return None
#         if len(x) == 0 or len(y) == 0 or len(x.shape) != 2 or len(y.shape) != 2:
#             return None
#         try:
#             x = self.add_intercept(x)
#             for iter_ in range(1, self.n_cycle + 1):
#                 gradients = (1 / len(x)) * np.dot(x.T,
#                                                   np.subtract(np.dot(x, self.theta), y))
#                 self.theta = np.subtract(self.theta, (gradients * self.alpha))
#         except:
#             print("Params error in fit_")
#             return None

#     def mse_(self, y, y_hat):
#         mse = ((y - y_hat) ** 2).sum()
#         return mse/y.shape[0]

#     @staticmethod
#     def select_color(color):
#         if color == 'green':
#             color = 'forestgreen'
#             l_color = 'lime'
#         elif color == 'purple':
#             color = 'darkmagenta'
#             l_color = 'violet'
#         else:
#             color = 'navy'
#             l_color = 'royalblue'
#         return color, l_color

#     @staticmethod
#     def plot_fit_data(x, y, y_hat, label_data='', color='', x_label=''):
#         color, l_color = MyLinearRegression.select_color(color)
#         label_pred = f'Predicted {label_data.lower()}'
#         fig = plt.figure()
#         plt.scatter(x, y, c=color, label=label_data)
#         plt.scatter(x, y_hat, c=l_color, label=label_pred, s=7)
#         plt.xlabel(f'{x_label}')
#         plt.ylabel('y: sell prince (in keuros)')
#         plt.legend()
#         plt.grid()

import numpy as np


class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas: np.ndarray, alpha: float = 0.001, max_iter: int = 1000):
        if isinstance(thetas, list):
            thetas = np.asarray(thetas).reshape(-1, 1)
        thetas = thetas.astype("float64")
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    @staticmethod
    def mse_(y: np.ndarray, y_hat: np.ndarray) -> float:
        if y.shape != y_hat.shape:
            return None
        mse_elem = (y_hat - y) ** 2 / (y.shape[0])
        return np.sum(mse_elem)

    @staticmethod
    def cost_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Description:
            Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
        Args:
            y: has to be an numpy.ndarray, a vector.
            y_hat: has to be an numpy.ndarray, a vector.
        Returns:
            J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
            None if there is a dimension matching problem between X, Y or theta.
        Raises:
            This function should not raise any Exception.
        """
        if y.shape != y_hat.shape:
            return None
        res = (y_hat - y) ** 2 / (2 * y.shape[0])
        return res

    @staticmethod
    def cost_(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Computes the half mean squared error of two non-empty numpy.ndarray,
            without any for loop. The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.ndarray, a vector.
            y_hat: has to be an numpy.ndarray, a vector.
        Returns:
            The half mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.ndarray.
            None if y and y_hat does not share the same dimensions.
        Raises:
            This function should not raise any Exceptions.
        """
        if y.shape != y_hat.shape:
            return None
        j_elem = MyLinearRegression.cost_elem_(y, y_hat)
        return np.sum(j_elem)

    @staticmethod
    def add_intercept(x: np.ndarray, axis: int = 1) -> np.ndarray:
        """Adds a column of 1's to the non-empty numpy.ndarray x.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
        Returns:
            X as a numpy.ndarray, a matrix of dimension m * (n + 1).
            None if x is not a numpy.ndarray.
            None if x is a empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        ones = np.ones((x.shape[0], 1))
        res = np.concatenate((ones, x), axis=axis)
        return res

    def predict_(self, x: np.ndarray) -> np.ndarray:
        """Computes the prediction vector y_hat from two non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
            theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exception.
        """
        theta = self.thetas
        if (x.shape[1] + 1) != theta.shape[0]:
            return None
        intercepted = self.add_intercept(x)
        y_hat = intercepted.dot(theta)
        return y_hat

    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes a gradient vector from three non-empty numpy.ndarray,
            without any for-loop. The three arrays must have the compatible dimensions.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
            theta: has to be an numpy.ndarray, a vector (n + 1) * 1.
        Returns:
            The gradient as a numpy.ndarray, a vector of dimensions n * 1,
            containg the result of the formula for all j.
            None if x, y, or theta are empty numpy.ndarray.
            None if x, y and theta do not have compatible dimensions.
        Raises:
            This function should not raise any Exception.
        """
        theta = self.thetas
        if (0 in [x.size, y.size, theta.size] or x.shape[0] != y.shape[0] or
                (x.shape[1] + 1) != theta.shape[0]):
            return None
        m = x.shape[0]
        y_hat = self.predict_(x)
        x = self.add_intercept(x)
        nabla_j = x.T.dot(y_hat - y) / m
        return nabla_j

    def fit_(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: a matrix of dimension m * n: (number of training examples, number of features).
            y: a vector of dimension m * 1: (number of training examples, 1).
            theta: a vector of dimension (n + 1) * 1: (number of features + 1, 1).
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
            new_theta: numpy.ndarray, a vector of dimension (number of features + 1, 1).
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        theta = self.thetas
        alpha = self.alpha
        if x.shape[0] != y.shape[0] or (x.shape[1] + 1) != theta.shape[0]:
            return None
        for _ in range(self.max_iter):
            new_theta = self.gradient_(x, y)
            theta -= alpha * new_theta
        self.thetas = theta
