import numpy as np


def gradient(x, y, theta):
    """"""
    parenthesis = np.subtract(x.dot(theta), y)
    coef = x.dot(1/x.shape[0])
    return np.transpose(coef).dot(parenthesis)


if __name__ == "__main__":
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])

    y = np.array([2, 14, -13, 5, 12, 4, -19])
    theta1 = np.array([3, 0.5, -6])
    print("# Example 0:")
    print(gradient(x, y, theta1))
    # Output:
    print("array([ -37.35714286, 183.14285714, -393. ])")
    print()

    print("# Example 1:")
    theta2 = np.array([0, 0, 0])
    print(gradient(x, y, theta2))
    # Output:
    print("array([ 0.85714286, 23.28571429, -26.42857143])")
