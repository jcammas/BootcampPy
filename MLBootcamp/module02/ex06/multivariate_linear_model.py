import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#   Fit a linear regression model to a dataset with multiple features.
#   Plot the model’s predictions and interpret the graphs

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


def draw_regression(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, feature: str,
                    colors: tuple, new_theta: np.ndarray, lr_model: MyLinearRegression):
    plt.xlabel(f"x: {feature}")
    plt.ylabel("y: sell price (in keuros)")
    plt.grid()

    plt.plot(x, y, "o", color=colors[0], label="Sell price")
    plt.plot(x, y_hat, "o", color=colors[1],
             label="Predicted sell price", markersize=3)
    plt.show()


# in the first part of the exercise, we will train three different univariate models to predict spaceships prices, this is univariate

def univariate_lr(data: pd.DataFrame, feature: str, colors: tuple,
                  theta: list = [0, 0], alpha: float = 0.001, max_iter: int = 800000) -> None:
    # your program has to perform a gradient descent from a new set of thetas
    linear_model = MyLinearRegression(theta, alpha, max_iter)
    x = np.array(data[feature]).reshape(-1, 1)
    y = np.array(data["Sell_price"]).reshape(-1, 1)
    linear_model.fit_(x, y)
    y_hat = linear_model.predict_(x)
    new_theta = linear_model.thetas
    # print the final value of the thetas
    print("new theta:", new_theta, sep='\n')
    # print the MSE of the corresponding model
    print("MyLR.mse_(y, y_hat):", linear_model.mse_(y, y_hat))
    # plot or generate a plot,
    draw_regression(x, y, y_hat, feature.lower(),
                    colors, new_theta, linear_model)


def multivariate_lr(data: pd.DataFrame,
                    theta: np.ndarray, alpha: float, max_iter: int = 800000) -> None:
    linear_model = MyLinearRegression(theta, alpha, max_iter)
    predicting_feature = "Sell_price"
    x = np.array(data.drop(predicting_feature, axis=1))
    y = np.array(data[predicting_feature]).reshape(-1, 1)

    linear_model.fit_(x, y)
    y_hat = linear_model.predict_(x)
    new_theta = linear_model.thetas
    print("new theta:", new_theta, sep='\n')
    print("MyLR.mse_(y, y_hat):", linear_model.mse_(y, y_hat))

    draw_regression(x[..., 0], y, y_hat, "age",
                    ("darkblue", "dodgerblue"), new_theta, linear_model)
    draw_regression(x[..., 1], y, y_hat, "thrust power",
                    ("g", "lime"), new_theta, linear_model)
    draw_regression(x[..., 2], y, y_hat, "terameters",
                    ("darkviolet", "violet"), new_theta, linear_model)


data = pd.read_csv("../resources/spacecraft_data.csv")


# univariate_lr(data, "Age", colors=("darkblue", "dodgerblue"),
#               theta=[1000.0, -1], alpha=2.5e-5)
# univariate_lr(data, "Thrust_power", colors=(
#     "g", "lime"), theta=[20, 5], alpha=0.0001)
# univariate_lr(data, "Terameters", colors=(
#     "darkviolet", "violet"), theta=[750, -3], alpha=0.0002)

multivariate_lr(data, theta=[1.0, 1.0, 1.0, 1.0],
                alpha=1e-6, max_iter=5000000)


# For each cases, verify the script generates the different corresponding graphs => we can verify this with the set up (color, ...) and it is the same as we can observe on the subject so its ok
# The program should print the values of vector thetas:
# - for Sell_price vs Age, theta should be closed to: [[647.04], [-12.99]] => it depends of the number of iteration and we are close, the more we increase iter, the closer we are
# - for Sell_price vs Thrust_power, theta should be closed to: [[39.88920787 4.32705362]] => perfect
# - for Sell_price vs Terameters, theta should be closed to: [[744.67913252 -2.86265137]] => perfect

# Verify theta vector is closed to: [[383.94598482 -24.29984158 5.67375056 -2.66542654]] => it depends of the number of iter, the more we have the longer is the compilation ...
# You should notice a sensible difference between the first three MSE and the fourth one (the 4th must be much smaller). YES oK
# - Verify the 3 graphs concerning the multivariate model are generated or can be visualized => yes we plot them
#  Student is able to explain how MSE result traduce the fact the predicted points are closer to the observed ones => multivariate does this, The smaller the mean squared error, the closer you are to finding the line of best fit. This is why we can relate our MSE result to our graph

# What can we say about the model ?
# First we can observe that the loss is lower when we use the multivariate model.
# Then we can observe that our prediction is better
# with univariate, we have an "average" for our predicted sell price => it is quite good but we have too much loss
# with multivariate, we are way more precise => it is very good and we don't have much loss
# disclaimer => becarefull of the overfitting (we will see this after with the data splitter)
