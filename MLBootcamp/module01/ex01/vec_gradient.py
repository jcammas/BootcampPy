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


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])

    theta1 = np.array([[2], [0.7]])
    print(gradient(x, y, theta1))

    theta2 = np.array([[1], [-0.4]])
    print(gradient(x, y, theta2))

    x = np.array([12.4956442, 21.5007972, 31.5527382,
                 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287,
                 46.6793434, 59.5585554]).reshape((-1, 1))
    print("# Example 0:")
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(gradient(x, y, theta1))
    # print(np.gradient(predict_(x, theta1), y))
    # Output:
    print("array([[-19.0342574], [-586.66875564]])")
    print()

    print("# Example 1:")
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(gradient(x, y, theta2))
    # Output:
    print("array([[-57.86823748], [-2230.12297889]])")

    def unit_test(n, theta, answer, f):
        x = np.array(range(1, n+1)).reshape((-1, 1))
        y = f(x)
        print(f"Student:\n{gradient(x, y, theta)}")
        print(f"Truth  :\n{answer}")
        print()

    theta = np.array([[1.], [1.]])
    answer = np.array([[-11.625], [-795.375]])
    unit_test(100, theta, answer, lambda x: 1.25 * x)

    answer = np.array([[-124.125], [-82957.875]])
    unit_test(1000, theta, answer, lambda x: 1.25 * x)

    answer = np.array([[-1.24912500e+03], [-8.32958288e+06]])
    unit_test(10000, theta, answer, lambda x: 1.25 * x)

    theta = np.array([[4], [-1]])
    answer = np.array([[-13.625], [-896.375]])
    unit_test(100, theta, answer, lambda x: -0.75 * x + 5)

    answer = np.array([[-126.125], [-83958.875]])
    unit_test(1000, theta, answer, lambda x: -0.75 * x + 5)

    answer = np.array([[-1.25112500e+03], [-8.33958388e+06]])
    unit_test(10000, theta, answer, lambda x: -0.75 * x + 5)
