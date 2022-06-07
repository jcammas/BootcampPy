import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


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


class Minmax():
    def __init__(self):
        self.min = 0.
        self.max = 0.

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def apply(self, X):
        e = 1e-20
        mnmx = (X - self.min) / (self.max - self.min + e)
        return mnmx

    def unapply(self, X):
        e = 1e-20
        return (X * (self.max - self.min + e)) + self.min


class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """

    def __init__(self, theta: np.ndarray, alpha: float = 1e-3, max_iter: int = 1000):
        if isinstance(theta, list):
            theta = np.asarray(theta).reshape(-1, 1)
        theta = theta.astype("float64")
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    @staticmethod
    def loss_(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15) -> float:
        """
        Computes the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
            y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
            eps: has to be a float, epsilon (default=1e-15)
        Returns:
            The logistic loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        if y.shape != y_hat.shape:
            return None
        ones = np.ones(y.shape)
        m = y.shape[0]
        res = np.sum(y * np.log(y_hat + eps) + (ones - y)
                     * np.log(ones - y_hat + eps)) / -m
        return res

    @staticmethod
    def sigmoid_(x: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid of a vector.
        Args:
        x: has to be an numpy.array, a vector
        Return:
        The sigmoid value as a numpy.array.
        None otherwise.
        Raises:
        This function should not raise any Exception.
        """
        # sigmoid(x) = 1 / (1 + e^−x)
        if x.size == 0:
            return None
        return 1 / (1 + np.exp(-x))

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
    def sigmoid_(x: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid of a vector.
        Args:
            x: has to be an numpy.ndarray, a vector
        Returns:
            The sigmoid value as a numpy.ndarray.
            None if x is an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        if x.size == 0:
            return None
        return 1 / (1 + np.exp(-x))

    def predict_(self, x: np.ndarray) -> np.ndarray:
        """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exception.
        """
        # yˆ = sigmoid(X'·θ) = 1 / (1 + e^−X'.·0)
        x_ = self.add_intercept(x)
        y = self.sigmoid_(x_.dot(self.theta))
        return y

    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes a gradient vector from three non-empty numpy.ndarray, without any a for-loop.
            The three arrays must have compatible dimensions.
        Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector (n + 1) * 1.
        Returns:
        The gradient as a numpy.ndarray, a vector of dimensions (n + 1) * 1, containing
            the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """
        theta = self.theta
        if (0 in [x.size, y.size, theta.size] or x.shape[0] != y.shape[0] or
                (x.shape[1] + 1) != theta.shape[0]):
            return None
        y_hat = self.predict_(x)
        x = self.add_intercept(x)
        res = x.T.dot(y_hat - y) / x.shape[0]
        return res

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
        for _ in range(self.max_iter):
            grad = self.gradient_(x, y).sum(axis=1)
            self.theta = self.theta - (self.alpha * grad).reshape((-1, 1))
        return self


def round_number_for_pred(X):
    """
    Here we want to facilitate the calcul in order to get 1 or 0 to know if our citizen's zipcode corresponds
    or not to our favorite planet according to our new criterion.
    Based on the selected Space Zipcode (0, 1, 2, 3), generate a new numpy.array to
    label each citizen according to your new selection criterion:
        • 1 if the citizen’s zipcode corresponds to your favorite planet.
        • 0 if the citizen has another zipcode.
    """
    Y_hat = mlr.predict_(X)
    # pour rappel on arrondi à l'entier supérieur à partir de 0.5
    Y_hat[Y_hat >= 0.5] = 1
    # pour rappel on arrondi à l'entier inférieur si on est à 0.4 ou moins
    Y_hat[Y_hat < 0.5] = 0
    return Y_hat


def correct_pred(mlr, X, Y):
    """
    Calculate and display the fraction of correct predictions over the total number of
    predictions based on the test set
    """
    Y_hat = round_number_for_pred(X)
    return np.sum(Y_hat == Y) / Y.shape[0]


def read_y_data(Y, feature):
    k = -1
    Y[(Y == feature)] = k
    Y[(Y != k)] = 0.
    Y[(Y == k)] = 1.
    return Y


def draw_mono_log(X, Y_hat, Y):
    """
    Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the
    final prediction of the model.
    """
    plot_dim = 2
    fig, axs_ = plt.subplots(plot_dim, plot_dim)
    axs = []
    for sublist in axs_:
        for item in sublist:
            axs.append(item)

    sns.scatterplot(ax=axs[0], x=X[:, 0].reshape(-1),
                    y=X[:, 1].reshape(-1), hue=Y_hat.reshape(-1))
    sns.scatterplot(ax=axs[1], x=X[:, 0].reshape(-1),
                    y=X[:, 2].reshape(-1), hue=Y_hat.reshape(-1))
    sns.scatterplot(ax=axs[2], x=X[:, 1].reshape(-1),
                    y=X[:, 2].reshape(-1), hue=Y_hat.reshape(-1))

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # here we need to pass as an argument the zipcode => and we need to display the usage if not
    parser.add_argument("-z",
                        "--zipcode",
                        help="Solar citizens origin area",
                        type=float,
                        choices=[0., 1., 2., 3.],
                        required=True)

    args = parser.parse_args()

    # You will work with data from the last Solar System Census.
    # The dataset is divided in two files which can be found in the resources folder:
    # solar_system_census.csv and solar_system_census_planets.csv.
    # The first file contains biometric information such as the height, weight, and bone
    # density of several Solar System citizens.
    # The second file contains the homeland of each citizen, indicated by its Space Zipcode
    # representation (i.e. one number for each planet... :)).

    data_x = pd.read_csv("../resources/solar_system_census.csv")
    data_y = pd.read_csv("../resources/solar_system_census_planets.csv")

    X = np.array(data_x[["height", "weight", "bone_density"]]).reshape(-1, 3)
    Y = np.array(data_y["Origin"]).reshape(-1, 1)

    minmax = Minmax()
    X = minmax.fit(X).apply(X)

    Y = read_y_data(Y, args.zipcode)

    # Split the dataset into a training and a test set
    X_train, X_test, Y_train, Y_test = data_spliter(X, Y, 0.8)

    theta = [0] * (X.shape[1] + 1)
    mlr = MyLogisticRegression(theta, alpha=1e-2, max_iter=100000)
    mlr.fit_(X_train, Y_train)

    acc = correct_pred(mlr, X_test, Y_test)
    print(
        f"fraction of correct predictions over the total number of predictions based on the test set : {acc}")

    Y_hat = round_number_for_pred(X)
    draw_mono_log(X, Y_hat, Y)
