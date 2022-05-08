import numpy as np


def add_intercept(x: np.ndarray) -> np.ndarray:
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
    try:
        shape = (x.shape[0], 1)
        ones = np.full(shape, 1)
        res = np.concatenate((ones, x), axis=1)
        return res
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
    i = add_intercept(x)
    if i.shape[1] != theta.shape[0]:
        return None
    return i.dot(theta)


def loss_elem_(theta, X, Y):
    """
    Description:
    Calculates all the elements 0.5*M*(y_pred - y)^2 of the cost
    function.
    Args:
    theta: has to be a numpy.ndarray, a vector of dimension (number of
    features + 1, 1).
    X: has to be a numpy.ndarray, a matrix of dimension (number of
    training examples, number of features).
    Returns:
    J_elem: numpy.ndarray, a vector of dimension (number of the training
    examples,1).
    None if there is a dimension matching problem between X, Y or theta.
    Raises:
    This function should not raise any Exception.
    """
    Y_hat = predict_(X, theta)
    if Y_hat is None:
        return None
    loss = np.zeros((len(Y), 1))
    for i in range(len(Y)):
        try:
            loss[i] = ((Y_hat[i] - Y[i])**2)
        except np.core._exceptions.UFuncTypeError:
            return None
    return loss


def loss_(theta, X, Y):
    """
    Description:
    Calculates the value of cost function.
    Args:
    theta: has to be a numpy.ndarray, a vector of dimension (number of
    features + 1, 1).
    X: has to be a numpy.ndarray, a vector of dimension (number of
    training examples, number of features).
    Returns:
    J_value : has to be a float.
    None if X does not match the dimension of theta.
    Raises:
    This function should not raise any Exception.
    """
    loss = loss_elem_(theta, X, Y)
    if loss is None:
        return None
    loss /= 2 * len(Y)
    return loss.sum()


X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
print(loss_elem_(theta1, X1, Y1), "\n")
print(loss_(theta1, X1, Y1),  "\n")


X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
Y2 = np.array([[19.], [42.], [67.], [93.]])
print(loss_elem_(theta2, X2, Y2),  "\n")
print(loss_(theta2, X2, Y2),  "\n")


X3 = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
theta3 = np.array([[0.], [1.]])
Y3 = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
print(loss_(theta3, X3, Y3),  "\n")
print(loss_(theta3, Y3, Y3),  "\n")


print(loss_(theta3, X3, X3))


y_hat = np.array([[1], [2], [3], [4]])
y = np.array([[0], [0], [0], [0]])

print()
print(loss_elem_(theta3, y, y_hat))
print(
    "ici le résultat est bon car il correspond à [[0.125], [0.5], [1.125], [2]] d'un pdv relation \n 0.125 => 1 \n 0.125*4 = 0.5 => 4 / 1 = 1 \n 0.5*2.25 = 1.125 => 9 / 4 = 2.25 \n 1.125 * 1.77 = 2 => 16 / 9 = 1.77")

print(loss_(theta3, y, y_hat))
print()
