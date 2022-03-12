import numpy as np
from tools import add_intercept


def gradient(x, y, theta):
    """"""
    x_bias = add_intercept(x)
    y_pred = np.dot(x_bias, theta)
    y_pred = y_pred.reshape(len(y), 1)
    y = y.reshape(len(y), 1)
    error = y_pred - y
    error_columns = error.reshape(len(y), 1)
    error_dot_x = np.dot(error_columns.T, x_bias)
    grad = 1/len(x) * error_dot_x
    return grad.reshape(len(theta),)


x = np.array([[12.4956442], [21.5007972], [
             31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [
             45.7655287], [46.6793434], [59.5585554]])

theta1 = np.array([[2], [0.7]])
print(gradient(x, y, theta1))

theta2 = np.array([[1], [-0.4]])
print(gradient(x, y, theta2))
