import numpy as np
from prediction import predict_


def simple_gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    if x.shape[1] != 1 or y.shape[1] != 1 or theta.shape != (2, 1):
        return None
    return np.array([1.0 / len(x) * np.sum(predict_(x, theta) - y), 1.0 / len(x) * np.sum((predict_(x, theta) - y) * x)])


x = np.array([[12.4956442], [21.5007972], [
             31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [
             45.7655287], [46.6793434], [59.5585554]])

theta1 = np.array([[2], [0.7]])
print(simple_gradient(x, y, theta1))

theta2 = np.array([[1], [-0.4]])
print(simple_gradient(x, y, theta2))
