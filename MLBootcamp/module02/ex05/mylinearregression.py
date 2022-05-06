import numpy as np
import warnings
import sys
warnings.filterwarnings('ignore')

ALLOWED_TYPES = [int, float, np.int64, np.float64]


class MyLinearRegression():

    def __init__(self, theta, alpha=0.001, n_cycle=2000):
        if not MyLinearRegression.verif_params(theta):
            sys.exit('Invalid theta param')
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.theta = np.asarray(theta)

    @staticmethod
    def verif_params(*args):
        for arg in args:
            if not isinstance(arg, list) and not isinstance(arg, np.ndarray):
                return False
            for val in arg:
                if type(val) not in ALLOWED_TYPES:
                    if isinstance(val, np.ndarray) or isinstance(val, list):
                        try:
                            tmp = val.sort()
                            for v in val:
                                if type(v) not in ALLOWED_TYPES:
                                    return False
                        except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
                            return False
                        continue
                    else:
                        return False
        return True

    def predict_(self, X):
        if self.theta.ndim != 2 or X.ndim != 2 or self.theta.shape[1] != 1 or X.shape[1] + 1 != self.theta.shape[0]:
            print("Incompatible dimension match between X and theta.")
            return None
        if not MyLinearRegression.verif_params(X):
            return None
        X = np.insert(X, 0, 1., axis=1)
        return X.dot(self.theta)

    def loss_elem_(self, X, Y):
        if not MyLinearRegression.verif_params(X, Y):
            return None
        Y_hat = self.predict_(X)
        res = np.array([(Y - Y_hat) ** 2])
        return res

    def loss_(self, X, Y):
        if not MyLinearRegression.verif_params(X, Y):
            return None
        Y_hat = self.predict_(X)
        if Y_hat is None:
            return None
        return np.sum((Y_hat - Y)**2)/(2*X.shape[0])

    def fit_(self, X, Y):
        if self.theta.ndim != 2 or X.ndim != 2 or self.theta.shape[1] != 1 or X.shape[1] + 1 != self.theta.shape[0] or Y.shape[0] != X.shape[0]:
            print("Incompatible dimension match between X and theta.")
            return None
        if not MyLinearRegression.verif_params(X, Y):
            return None
        if len(X) == 0 or len(Y) == 0 or len(X.shape) != 2 or len(Y.shape) != 2:
            return None
        if not isinstance(self.alpha, float) or not isinstance(self.n_cycle, int):
            return None
        m = X.shape[0]
        X = np.insert(X, 0, 1., axis=1)
        for i in range(self.n_cycle):
            hypothesis = X.dot(self.theta)
            parenthesis = np.subtract(hypothesis, Y)
            sigma = np.sum(np.dot(X.T, parenthesis), keepdims=True, axis=1)
            self.theta = self.theta - (self.alpha / m) * sigma
        return self.theta
