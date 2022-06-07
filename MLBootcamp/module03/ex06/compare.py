import sys
import numpy as np

inf_lim = -600
sup_lim = 256


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        # Checking of the attributes:
        if (not isinstance(theta, (np.ndarray, tuple, list))) \
            or (not isinstance(alpha, (int, float))) \
                or (not isinstance(max_iter, int)):
            s = "At least one of the parameters is not of expected type."
            raise TypeError(s)

        # Conversion of thetas and testing the shape of the parameters.
        theta = self._convert_thetas_(theta)
        if (alpha >= 1) or (alpha <= 0) or (max_iter <= 0):
            return None

        # Casting self.theta to float, in case it is integer
        self.theta = theta.astype('float64')
        self.alpha = float(alpha)
        self.max_iter = max_iter

    @staticmethod
    def _convert_thetas_(theta):
        """ Private function, convert theta parameter in the constructor
        from tuple or list into numpy ndarray.
        Args:
            theta: list, tuple or numpy ndarray containing the model's
                    coefficients.
        """
        if isinstance(theta, np.ndarray):
            return theta
        return np.array(theta).reshape(-1, 1)

    def _sigmoid_(self, x):
        """ Compute the sigmoid of a vector.
        Args:
            x: has to be an numpy.array, a vector
        Return:
            The sigmoid value as a numpy.array.
            None otherwise.
        Raises:
            This function should not raise any Exception.
        """
        try:
            x_ = np.array(x, copy=True)
            x_[x_ < inf_lim] = inf_lim
            x_[x_ > sup_lim] = sup_lim
            return np.divide(1., 1. + np.exp(-x))
        except:
            return None

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of shape m * n.
            theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
        Return:
            y_hat: a numpy.array of shape m * 1, when x and theta numpy arrays
                with expected and compatible shapes.
            None: otherwise.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if (not isinstance(x, np.ndarray)):
                s = "x is not of the expected type (numpy array)."
                print(s, file=sys.stderr)
                return None

        # Checking the shape of x and y
            if x.ndim != 2 or \
                    (x.shape[1] + 1 != self.theta.shape[0]):
                s = "x is not 2 dimensional array " \
                    + "or mismatching shape between x and self.theta"
                print(s, file=sys.stderr)
                return None

            x_ = np.hstack((np.ones((x.shape[0], 1)), x))
            ypred = self._sigmoid_(np.dot(x_, self.theta))
            return ypred
        except:
            return None

    def _gradient_(self, x, y):
        """ Private function gradient, there is no test performed on the
        parameters. It is to avoid to perform useless same tests at every
        iteration of the loop in the fit method.
        Args:
            x: has to be an numpy.array, a matrix of shape m * n.
            y: has to be an numpy.array, a vector of shape m * 1.
        Return:
            The gradient as a numpy.array, a vector of shape n * 1,
        """
        xp = np.hstack((np.ones((x.shape[0], 1)), x))
        return xp.T @ (xp @ self.theta - y) / x.shape[0]

    def gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array,
        without any for-loop. The three arrays must have compatible shapes.
        Args:
            x: has to be an numpy.array, a matrix of shape m * n.
            y: has to be an numpy.array, a vector of shape m * 1.
            theta: has to be an numpy.array, a vector (n +1) * 1.
        Return:
            The gradient as a numpy.array, a vector of shape n * 1,
                containing the result of the formula for all j.
            None if x, y, or theta are empty numpy.array.
            None if x, y and theta do not have compatible shapes.
            None if y, x or theta is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if (not isinstance(x, np.ndarray)) \
                    or (not isinstance(y, np.ndarray)):
                s = "x or/and y or/and theta are not of the expected type" \
                    + " (numpy array)."
                print(s, file=sys.stderr)
                return None

            # Checking the shape of x and y
            if (y.ndim != 2) or (x.ndim != 2) \
                    or (y.shape[1] != 1) \
                    or (y.shape[0] != x.shape[0]) \
                    or (self.theta.shape != (x.shape[1] + 1, 1)):
                s = "Unexpected dimension for at least one of the arrays" \
                    + " or mismatching shape between arrays"
                print(s, file=sys.stderr)
                return None

            grad = self._gradient_(x, y)
            return grad
        except:
            return None

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.array, a vector of shape m * 1:
               (number of training examples, 1).
            y: has to be a numpy.array, a vector of shape m * 1:
               (number of training examples, 1).
        Return:
            self: instance of MyLogisiticRegression.
            None if there is a matching shape problem.
            None if x or y is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # Checking x, y and theta are numpy array
            if (not isinstance(x, np.ndarray)) \
                    or (not isinstance(y, np.ndarray)):
                s = "Unexpected type for one of the array."
                print(s, file=sys.stderr)
                return None

            # Checking the shape of x and y
            if (y.ndim != 2) or (x.ndim != 2) \
                    or (y.shape[1] != 1) \
                    or (y.shape[0] != x.shape[0]) \
                    or (self.theta.shape != (x.shape[1] + 1, 1)):
                s = "Unexpected dimension for at least one of the arrays" \
                    + " or mismatching shape between arrays"
                print(s, file=sys.stderr)
                return None

            # Performing the gradient descent
            for _ in range(self.max_iter):
                grad = self._gradient_(x, y)
                self.theta = self.theta - self.alpha * grad
            return self
        except:
            # If something unexpected happened, we juste leave
            print("Something wrong during fit.", file=sys.stderr)
            return None

    def loss_elem_(self, x, y):
        """
        Computes the logistic loss vector.
        Args:
            x: has to be an numpy.array, a vector of shape m * n.
            y: has to be an numpy.array, a vector of shape m * 1.
        Return:
            The logistic loss vector numpy.ndarray.
            None otherwise.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # Checking x and y are numpy array
            if (not isinstance(x, np.ndarray)) \
                    or (not isinstance(y, np.ndarray)):
                return None

            # Checking the shape of x and y
            if (y.ndim != 2) or (x.ndim != 2) \
                    or (y.shape[1] != 1) \
                    or (y.shape[0] != x.shape[0]) \
                    or (self.theta.shape != (x.shape[1] + 1, 1)):
                s = "Unexpected dimension for at least one of the arrays" \
                    + " or mismatching shape between arrays"
                print(s, file=sys.stderr)
                return None

            eps = 1e-30
            yhat = self.predict_(x)
            #log_loss = y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps)
            log_loss = self._loss_elem_(y, yhat)
            return log_loss
        except:
            return None

    def loss_(self, x, y):
        """Computes the logistic loss value.
        Args:
            x: has to be an numpy.array, a vector.
            y: has to be an numpy.array, a vector.
        Returns:
            The logistic loss value as a float.
            None if y or y_hat are empty numpy.array.
            None if x and y does not share the same dimensions.
            None if x or y is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # Checking y and y_hat are numpy array
            if (not isinstance(x, np.ndarray)) \
                    or (not isinstance(y, np.ndarray)):
                return None

            # Checking the shape of x and y
            if (y.ndim != 2) or (x.ndim != 2) \
                    or (y.shape[1] != 1) \
                    or (y.shape[0] != x.shape[0]) \
                    or (self.theta.shape != (x.shape[1] + 1, 1)):
                s = "Unexpected dimension for at least one of the arrays" \
                    + " or mismatching shape between arrays"
                print(s, file=sys.stderr)
                return None

            log_loss = np.mean(self.loss_elem_(x, y))
            return log_loss
        except:
            return None

    @staticmethod
    def _loss_elem_(y, y_hat):
        """Computes the logistic loss vector.
        Args:
            y: has to be an numpy.array, a vector of shape m * 1.
            y_hat: has to be an numpy.array, a vector of shape m * 1.
        Return:
            The logistic loss vector numpy.ndarray.
            None otherwise.
        Raises:
            This function should not raise any Exception.
        """
        try:
            eps = 1e-30
            log_loss = y * np.log(y_hat + eps) + (1 - y) * \
                np.log(1 - y_hat + eps)
            return log_loss
        except:
            return None

    @staticmethod
    def _loss_(y, y_hat):
        """Private method to compute the logistic loss value.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            The logistic loss value as a float.
            None otherwise (type, shape, dimension issue ...).
        Raises:
            This function should not raise any Exception.
        """
        try:
            # Checking y and y_hat are numpy array
            if (not isinstance(y, np.ndarray)) \
                    or (not isinstance(y_hat, np.ndarray)):
                return None

            # Checking the shape of y and y_hat
            if (y.shape[1] != 1) \
                or (y_hat.shape[1] != 1) \
                    or (y_hat.shape[0] != y.shape[0]):
                return None
            log_loss = MyLogisticRegression._loss_elem_(y, y_hat)
            return np.mean(log_loss)
        except:
            # If something unexpected happened, we juste leave
            return None


if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])

    mylr = MyLogisticRegression([2, 0.5, 7.1, -4.3, 2.09])

    print("# Example 0:")
    my_res = mylr.predict_(X)
    # Output:
    expected = np.array([[0.99930437], [1.], [1.]])
    print("my prediction:".ljust(25), my_res.reshape(1, -1))
    print("expected prediction:".ljust(25), my_res.reshape(1, -1))

    print("# Example 1:")
    my_res = mylr.loss_(X, Y)
    # Output:
    expected = 11.513157421577004
    print("my loss:".ljust(25), my_res.reshape(1, -1))
    print("expected loss:".ljust(25), my_res.reshape(1, -1))

    print("# Example 2:")
    mylr.fit_(X, Y)
    my_res = mylr.theta
    # Output:
    expected = np.array([[1.04565272],
                         [0.62555148],
                         [0.38387466],
                         [0.15622435],
                         [-0.45990099]])
    print("my theta after fit:".ljust(25), my_res.reshape(1, -1))
    print("expected theta after fit:".ljust(25), my_res.reshape(1, -1))

    print("# Example 3:")
    my_res = mylr.predict_(X)
    # Output:
    expected = np.array([[0.72865802],
                         [0.40550072],
                         [0.45241588]])
    print("my prediction:".ljust(25), my_res.reshape(1, -1))
    print("expected prediction:".ljust(25), my_res.reshape(1, -1))

    print("# Example 4:")
    my_res = mylr.loss_(X, Y)
    # Output:
    expected = 0.5432466580663214
    print("my loss:".ljust(25), my_res.reshape(1, -1))
    print("expected loss:".ljust(25), my_res.reshape(1, -1))
