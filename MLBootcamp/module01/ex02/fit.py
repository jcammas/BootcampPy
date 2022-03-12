import numpy as np
from prediction import predict_


def fit_(theta, X, Y, alpha=0.001, max_iter=10000):
    if theta.ndim != 2 or X.ndim != 2 or theta.shape[1] != 1 or X.shape[1] + 1 != theta.shape[0] or Y.shape[0] != X.shape[0]:
        return None
    m = X.shape[0]
    X = np.insert(X, 0, 1., axis=1)
    for i in range(max_iter):
        hypothesis = X.dot(theta)
        parenthesis = np.subtract(hypothesis, Y)
        sigma = np.sum(np.dot(X.T, parenthesis), keepdims=True, axis=1)
        theta = theta - (alpha / m) * sigma
    return theta


x = np.array([[12.4956442], [21.5007972], [
             31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [
             45.7655287], [46.6793434], [59.5585554]])
theta1 = np.array([[1], [1]])
theta1 = fit_(theta1, x, y, alpha=5e-6, max_iter=15000)
print(theta1)
print("\n\n")
print(predict_(x, theta1))
