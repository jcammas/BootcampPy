import numpy as np
from prediction import predict_


def fit_(theta, x, y, alpha, max_iter):
    if theta.ndim != 2 or x.ndim != 2 or theta.shape[1] != 1 or x.shape[1] + 1 != theta.shape[0] or y.shape[0] != x.shape[0]:
        return None
    m = x.shape[0]
    x = np.insert(x, 0, 1., axis=1)
    for i in range(max_iter):
        hypothesis = x.dot(theta)
        parenthesis = np.subtract(hypothesis, y)
        sigma = np.sum(np.dot(x.T, parenthesis), keepdims=True, axis=1)
        theta = theta - (alpha / m) * sigma
    return theta


x = np.array([[0.2, 2., 20.], [0.4, 4., 40.],
              [0.6, 6., 60.], [0.8, 8., 80.]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.], [1.], [1.], [1.]])
print("# Example 0:")
theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
print(theta2)


print("# Example 1:")
print(predict_(x, theta2))

# print("CORRECTION: ")
# x = np.arange(1, 13).reshape(-1, 3)
# y = np.arange(9, 13).reshape(-1, 1)
# theta = np.array([[5], [4], [-2], [1]])
# alpha = 1e-2
# max_iter = 10000
# print(f"{fit_(x, y, theta, alpha = alpha, max_iter=max_iter)}")
# print(f"Answer = array([[ 7.111..],[ 1.0],[-2.888..],[ 2.222..]])")

# x = np.arange(1, 31).reshape(-1, 6)
# theta = np.array([[4], [3], [-1], [-5], [-5], [3], [-2]])
# y = np.array([[128], [256], [384], [512], [640]])
# alpha = 1e-4
# max_iter = 42000
# print(f"{fit_(x, y, theta, alpha=alpha, max_iter=max_iter)}")
# print(f"""Answer = array([[ 7.01801797]
#     [ 0.17717732]
#     [-0.80480472]
#     [-1.78678675]
#     [ 1.23123121]
#     [12.24924918]
#     [10.26726714]])""")
