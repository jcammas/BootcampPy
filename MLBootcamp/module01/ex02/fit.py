import numpy as np


def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    Returns:
    x as a numpy.array, a vector of shape m * 2.
    None if x is not a numpy.array.
    None if x is a empty numpy.array.
    Raises:
    This function should not raise any Exception"""
    if not isinstance(x, np.ndarray):
        return None
    if len(x.shape) != 2 or x.shape[1] != 1:
        return None
    try:
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        i = np.ones((x.shape[0], 1))
        return np.append(i, x, axis=1)
    except ValueError:
        return None


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of shape m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if add_intercept(x).shape[1] != theta.shape[0]:
        return None
    return np.dot(add_intercept(x), theta)


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be a numpy.array, a matrix of shape m * 1.
    y: has to be a numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta is an empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception."""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    if x.shape[1] != 1 or y.shape[1] != 1 or theta.shape != (2, 1):
        return None
    try:
        m = x.shape[0]
        x = add_intercept(x)
        res = x.T.dot(x.dot(theta) - y)
    except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
        return None
    return res / m


def fit_(theta, x, y, alpha=0.001, max_iter=10000):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
    y: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
    theta: has to be a numpy.array, a vector of shape 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of shape 2 * 1.
    None if there is a matching shape problem.
    None if x, y, theta, alpha or max_iter is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray) or\
            not isinstance(alpha, float) or not isinstance(max_iter, int):
        return None
    if x.shape[1] != 1 or y.shape[1] != 1 or theta.shape != (2, 1):
        return None
    if max_iter <= 0:
        return None
    theta = theta.astype("float64")
    while max_iter > 0:
        # repeat until convergence: {
        #      compute ∇(J)
        #      θ0 := θ0 − α∇(J)0
        #      θ1 := θ1 − α∇(J)1
        #  }
        #  Where:
        #     • α (alpha) is the learning rate. It’s a small float number (usually between 0 and 1),
        #     • For now, "reapeat until convergence" will mean to simply repeat for max_iter (a
        #       number that you will choose wisely)
        new_theta = gradient(x, y, theta)
        theta[0][0] -= alpha * new_theta[0][0]
        theta[1][0] -= alpha * new_theta[1][0]
        max_iter -= 1
    return theta


if __name__ == "__main__":

    x = np.array([[12.4956442], [21.5007972], [
        31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
        45.7655287], [46.6793434], [59.5585554]])
    theta1 = np.array([[1], [1]])
    theta1 = fit_(theta1, x, y, alpha=5e-6, max_iter=15000)
    print(theta1)
    print("\n\n")
    print(predict_(x, theta1))

    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([[1], [1]])
    print("# Example 0:")
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1)
    # Output:
    print("""
    array([[1.40709365],
    [1.1150909 ]])
    """)
    print()

    print("# Example 1:")
    print(predict_(x, theta1))
    # Output:
    print("""
    array([[15.3408728 ],
    [25.38243697],
    [36.59126492],
    [55.95130097],
    [65.53471499]])
    """)

    x = np.array(range(1, 101)).reshape(-1, 1)
    y = 0.75*x + 5
    theta = np.array([[1.], [1.]])
    print(fit_(x, y, theta, 5e-4, 20000))
    print("[[4.03112103], [0.76446193]]")

    # - with x = np.array(range(1,101)).reshape(-1,1),
    # y = 0.75*x + 5 and
    # theta = np.array([[1.],[1.]])
    # fit_(x, y, theta, 1e-5, 2000)
    # should return a result close to
    # [[4.03112103], [0.76446193]].
