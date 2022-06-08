import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class MyLinearRegression():
    """"""

    def __init__(self, thetas: np.ndarray, alpha: float = 0.001, max_iter: int = 1000):
        if isinstance(thetas, list):
            thetas = np.asarray(thetas).reshape(-1, 1)
        thetas = thetas.astype("float64")
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

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

    @staticmethod
    def mse_(y: np.ndarray, y_hat: np.ndarray) -> float:
        if y.shape != y_hat.shape:
            return None
        mse_elem = (y_hat - y) ** 2 / (y.shape[0])
        return np.sum(mse_elem)

    @staticmethod
    def loss_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
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
        if len(y) == 0 or len(y_hat) == 0:
            return None
        try:
            def loss_func(y, y_, m): return (y - y_) ** 2
            res = np.array([loss_func(i, j, len(y)) for i, j in zip(y, y_hat)])
            return res
        except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
            return None

    @staticmethod
    def loss_(y: np.ndarray, y_hat: np.ndarray) -> float:
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
        if len(y) == 0:
            return None
        try:
            res = abs((1 / (2 * y.shape[0])) *
                      (y_hat - y).T.dot(y - y_hat).sum())
            return res
        except (np.core._exceptions.UFuncTypeError, TypeError):
            return None

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
        thetas = self.thetas
        if (x.shape[1] + 1) != thetas.shape[0]:
            return None
        t = self.add_intercept(x)
        y_hat = t.dot(thetas)
        return y_hat

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        tmp = (x.dot(self.thetas))
        loss = tmp - y
        res = (x.T.dot(loss)) / x.shape[0]
        return res

    def fit_(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Description:
                Fits the model to the training dataset contained in x and y.
        Args:
                x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
                        examples, 1).
                y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
                        examples, 1).
                theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
                alpha: has to be a float, the learning rate
                max_iter: has to be an int, the number of iterations done during the gradient
                        descent
        Returns:
                new_theta: numpy.ndarray, a vector of dimension 2 * 1.
                None if there is a matching dimension problem.
        Raises:
                This function should not raise any Exception.
        """
        x_ = self.add_intercept(x)
        for _ in range(self.max_iter):
            swp = self.gradient(x_, y).sum(axis=1)
            tmp = (swp * self.alpha).reshape((-1, 1))
            self.thetas = self.thetas - tmp
        return self.thetas


class utils():
    def __init__(self):
        self.min = 0.
        self.max = 0.

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def apply(self, X):
        res = (X - self.min) / (self.max - self.min + 1e-20)
        return res


def add_polynomial_features(x, power):
    # Petit rappel : La régression linéaire est une régression polynomiale de degré 1.
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


def data_spliter(x, y, proportion):
    # We are using our data_spliter in order to create a training set and a test set
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


def model_save(data, model):
    """In models.[csv/yml/pickle] one must find the parameters of all the
    models you have explored and trained. In space_avocado.py train the model based on
    the best hypothesis you find and load the other models from models.[csv/yml/pickle].
    Then evaluate and plot the different graphics as asked before.
    https://www.journaldev.com/15638/python-pickle-example
    '"""
    path = os.path.join(os.path.dirname(__file__), f"model_{model}.yml")
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def model_load(model):
    """In models.[csv/yml/pickle] one must find the parameters of all the
    models you have explored and trained. In space_avocado.py train the model based on
    the best hypothesis you find and load the other models from models.[csv/yml/pickle].
    Then evaluate and plot the different graphics as asked before.
    https://www.journaldev.com/15638/python-pickle-example
    '"""
    path = os.path.join(os.path.dirname(__file__), f"model_{model}.yml")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# def plot_model(X, predicted_price, avocado_price, loss):
#     """"
#     plot our model in order to materialize our data
#     https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166"""
#     ax = plt.figure().add_subplot(projection='3d')

#     ax.scatter(X[:, 1], X[:, 2], avocado_price,
#                marker="*", c="g", label="Avocado price")
#     for i, y_hat in enumerate(predicted_price):
#         ax.scatter(X[:, 1], X[:, 2], y_hat, marker=["o", "s", "+", "*"][i],
#                    c=['r', 'y', 'm', 'b'][i], label=f"model {i}")
#     ax.legend()
#     plt.show()


def build_model(X, Y, model):
    print(f"model {model}")
    # Among the models trained, some must be based on polynomial features
    X_model = add_polynomial_features(X, model)

    # Regressions are made on a subset (training set) of the dataset
    # si on compile plusieurs fois on pourra observer que les données sont différentes => ça veut bien dire qu'on est sur un test et pas sur un entrainement
    X_train, X_test, Y_train, Y_test = data_spliter(X_model, Y, 0.8)

    # on défini notre theta pour initialiser notre régression linéaire
    theta = [0] * (model * X.shape[1] + 1)
    alpha = 1e-2

    lr = MyLinearRegression(thetas=theta, alpha=alpha, max_iter=100000)

    # The evaluation of the models must be made on the test sets.
    lr.fit_(X_test, Y_test)
    # Models are saved into a file(csv, yaml or pickle).
    model_save(lr, model)
    loss = lr.loss_(Y_test, lr.predict_(X_test))
    print(f"{loss = }")
    return loss, lr, lr.predict_(X_model)


def main():
    df = pd.read_csv("../resources/space_avocado.csv")

    X = np.array(
        df[["weight", "prod_distance", "time_delivery"]]).reshape(-1, 3)
    Y = np.array(df["target"]).reshape(-1, 1)

    std_X = utils()
    std_X.fit(X)
    X_ = std_X.apply(X)

    loss = []
    predictions = []
    for i in range(1, 5):
        l, lr, pred = build_model(X_, Y, model=i)
        loss.append(l)
        predictions.append(pred)

    # plot_model(X, predictions, Y, loss)


if __name__ == "__main__":
    main()
