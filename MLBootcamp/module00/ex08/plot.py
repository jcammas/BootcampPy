import numpy as np
from matplotlib import pyplot as plt

from prediction import predict_
from vec_loss import loss_


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    plt.plot(x, y, 'o')

    plt.plot(x, theta[1] * x + theta[0])
    p = predict_(x, theta)
    loss = loss_(y, p)

    plt.title("Cost: %f" % loss)
    for x_i, p_i, y_i in zip(x, p, y):
        plt.plot([x_i, x_i], [y_i, p_i], 'r--')
    plt.show()


x = np.arange(1, 6).reshape(-1, 1)
y = np.array([[11.52434424], [10.62589482], [
             13.14755699], [18.60682298], [14.14329568]])


theta1 = np.array([[18], [-1]])
plot_with_loss(x, y, theta1)

theta2 = np.array([[14], [0]])
plot_with_loss(x, y, theta2)

theta3 = np.array([[12], [0.8]])
plot_with_loss(x, y, theta3)
