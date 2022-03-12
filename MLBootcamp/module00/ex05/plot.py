import numpy as np
import matplotlib.pyplot as plt

from prediction import predict_


def plot(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> None:
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
    l = np.linspace(np.min(x), np.max(x), 2)
    plt.plot(l, predict_(l, theta), color="orange")
    plt.show()


x = np.arange(1, 6).reshape(-1, 1)
y = np.array([[3.74013816], [3.61473236], [
             4.57655287], [4.66793434], [5.95585554]])

theta1 = np.array([[4.5], [-0.2]])
plot(x, y, theta1)

theta2 = np.array([[-1.5], [2]])
plot(x, y, theta2)

theta3 = np.array([[3], [0.3]])
plot(x, y, theta3)
