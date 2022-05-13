import numpy as np


# with gradient calcul, we want to improve our model
# But how to get closer to the minimum? (in order to have the minimul loss we can)
# which direction =>  If the slope is positive, θ1 must be decreased. If the slope is negative, it must be increased


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
        shape = (x.shape[0], 1)
        ones = np.full(shape, 1)
        res = np.concatenate((ones, x), axis=1)
        return res
    except:
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


def simple_gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    if x.shape[1] != 1 or y.shape[1] != 1 or theta.shape != (2, 1):
        return None
    res = np.zeros(shape=(theta.shape))
    m = x.shape[0]
    y_hat = predict_(x, theta)
    for i in range(m):
        res[0][0] += (y_hat[i][0] - y[i][0])
        res[1][0] += (y_hat[i][0] - y[i][0]) * (x[i][0])
    return res / m


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])
    # Example 0:
    theta1 = np.array([[2], [0.7]])
    print(simple_gradient(x, y, theta1))

    # Example 1:
    theta2 = np.array([[1], [-0.4]])
    print(simple_gradient(x, y, theta2))

    x = np.array([[12.4956442], [21.5007972], [
        31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
        45.7655287], [46.6793434], [59.5585554]])

    theta1 = np.array([[2], [0.7]])
    print(simple_gradient(x, y, theta1))

    theta2 = np.array([[1], [-0.4]])
    print(simple_gradient(x, y, theta2))

    x = np.array([12.4956442, 21.5007972, 31.5527382,
                 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287,
                 46.6793434, 59.5585554]).reshape((-1, 1))
    print("# Example 0:")
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(simple_gradient(x, y, theta1))
    # Output:
    print("array([-19.0342574, -586.66875564])")
    print()

    print("# Example 1:")
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(simple_gradient(x, y, theta2))
    # Output:
    print("array([-57.86823748, -2230.12297889])")

    x = np.array(range(1, 11)).reshape((-1, 1))
    y = 1.25*x

    theta = np.array([[1.], [1.]])
    print(f"Student:\n{simple_gradient(x, y, theta)}")
    print(f"Truth  :\n{np.array([[-0.375],[-4.125]])}")
    print()

    theta = np.array([[1.], [-0.4]])
    print(f"Student:\n{simple_gradient(x, y, theta)}")
    print(f"Truth  :\n{np.array([[-8.075],[-58.025]])}")
    print()

    theta = np.array([[0.], [1.25]])
    print(f"Student:\n{simple_gradient(x, y, theta)}")
    print(f"Truth  :\n{np.array([[0.],[0.]])}")
    print()
