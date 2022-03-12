import numpy as np
from prediction import predict_ as predict


def loss_elem_(y, y_hat):
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
    return (y_hat - y) ** 2 / (2 * y.shape[0])


def loss_(y, y_hat):
    """
    Description:
    Calculates the value of cost function.
    Args:
    y: has to be an numpy.ndarray, a vector.
    y_hat: has to be an numpy.ndarray, a vector.
    Returns:
    26
    J_value : has to be a float.
    None if there is a dimension matching problem between X, Y or theta.
    Raises:
    This function should not raise any Exception.
    """
    if loss_elem_(y, y_hat) is None:
        return None
    return np.sum(loss_elem_(y, y_hat))


x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

print(loss_elem_(y1, y_hat1))
print(loss_(y1, y_hat1))
print("")
print("")

x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
y_hat2 = predict(x2, theta2)
y2 = np.array([[19.], [42.], [67.], [93.]])

print(loss_elem_(y2, y_hat2))
print("")
print(loss_(y2, y_hat2))
print("")

x3 = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
theta3 = np.array([[0.], [1.]])
y_hat3 = predict(x3, theta3)
y3 = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

print("")
print(loss_(y3, y_hat3))
print("")
print(loss_(y3, y3))
