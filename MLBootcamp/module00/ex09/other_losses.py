import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def mse_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape or y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    summed = 0.0
    for yi, yi_hat in zip(y, y_hat):
        summed += (yi - yi_hat) ** 2
    return summed/y.size


def rmse_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape or y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    if mse_(y, y_hat) is None:
        return None
    return mse_(y, y_hat) ** 0.5


def dot(x, y):
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        return None
    dot_product = 0.0
    for xi, yi in zip(x, y):
        dot_product += (xi * yi) ** 0.5
    return dot_product


def mae_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape or y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    return dot(y_hat - y, y_hat - y)/y.size


def r2score_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape or y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    try:
        return 1.0 - np.sum((y_hat - y)**2) / np.sum((y_hat - y.mean())**2)
    except:
        return None


x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

print("--- MSE ---\n")
print(mse_(x, y))
print(mean_squared_error(x, y))
print("\n")

print("--- RMSE ---\n")
print(rmse_(x, y))
print(sqrt(mean_squared_error(x, y)))
print("\n")

print("--- MAE ---\n")
print(mae_(x, y))
print(mean_absolute_error(x, y))
print("\n")

print("--- r2score ---\n")
print(r2score_(x, y))
print(r2_score(x, y))
print("\n")
