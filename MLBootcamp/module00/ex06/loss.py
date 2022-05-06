import numpy as np
from prediction import predict_ as predict


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
    Y_hat = predict(X, theta)
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
