import numpy as np

# this function is also known as Standard Logistic Sigmoid Function => logistic regression
# this function transforms an input into a probability value between 0 and 1 => use to classify the inputs


def sigmoid_(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be an numpy.array, a vector
    Return:
    The sigmoid value as a numpy.array.
    None otherwise.
    Raises:
    This function should not raise any Exception.
    """
    # sigmoid(x) = 1 / (1 + e^−x)

    if x.size == 0:
        return None
    return 1 / (1 + np.exp(-x))


print("\n\SUJET : \n")


# Example 1:
x = np.array(-4)
print(sigmoid_(x))
print("\ndoit retourner => [0.01798620996209156]\n")

# Example 2:
x = np.array(2)
print(sigmoid_(x))
print("\ndoit retourner => [0.8807970779778823]\n")

# Example 3:
x = np.array([[-4], [2], [0]])
print(sigmoid_(x))
print(
    "\ndoit retourner => [[0.01798620996209156], [0.8807970779778823], [0.5]]\n")

print("\n\nCORRECTION : \n")

x = np.array([0])
print(sigmoid_(x))
print("\ndoit retourner => [0.5]\n")

x = np.array([1])
print(sigmoid_(x))
print("\ndoit retourner => [0.73105857863]\n")

x = np.array([-1])
print(sigmoid_(x))
print("\ndoit retourner => [0.26894142137]\n")

x = np.array([50])
print(sigmoid_(x))
print("\ndoit retourner => [1]\n")

x = np.array([-50])
print(sigmoid_(x))
print("\ndoit retourner => [1.928749847963918e-22]\n")

x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
print(sigmoid_(x))
print(
    "\ndoit retourner => [0.07585818, 0.18242552, 0.37754067, 0.62245933, 0.81757448, 0.92414182]\n")
