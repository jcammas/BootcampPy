import numpy as np
from sklearn.metrics import confusion_matrix

# Compute confusion matrix to evaluate the accuracy of a classification.


def confusion_matrix_(y, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    labels: optional, a list of labels to index the matrix.
    This may be used to reorder or select a subset of labels. (default=None)
    df_option: optional, if set to True the function will return a pandas DataFrame
    instead of a numpy array. (default=False)
    Return:
    The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
    None if any error.
    Raises:
    This function should not raise any Exception.
    """
    if labels is None:
        labels = np.concatenate((y, y_hat))
        labels = np.unique(labels)
        labels = np.sort(labels)
    conf_matrix = np.zeros((len(labels), len(labels)))
    for i in range(len(labels)):
        for j in range(len(labels)):
            conf_matrix[i][j] = ((y == labels[i]) & (y_hat == labels[j])).sum()
    return conf_matrix


y_hat = np.array(["norminet", "dog", "norminet", "norminet", "dog", "bird"])
y = np.array(["dog", "dog", "norminet", "norminet", "dog", "norminet"])
# Example 1:
# your implementation
print(confusion_matrix_(y, y_hat))
# Output:
# array([[0 0 0]
#        [0 2 1]
#        [1 0 2]])
# sklearn implementation
print(confusion_matrix(y, y_hat))
# Output:
# array([[0 0 0]
#        [0 2 1]
#        [1 0 2]])
# Example 2:
# your implementation
print(confusion_matrix_(y, y_hat, labels=["dog", "norminet"]))
# Output:
# array([[2 1]
#        [0 2]])
# sklearn implementation
print(confusion_matrix(y, y_hat, labels=["dog", "norminet"]))
# Output:
# array([[2 1]
#        [0 2]])
# Example 3:
print(confusion_matrix_(y, y_hat, df_option=True))
# Output:
# bird dog norminet
# bird 0 0 0
# dog 0 2 1
# norminet 1 0 2
# Example 4:
print(confusion_matrix_(y, y_hat, labels=["bird", "dog"], df_option=True))
# Output:
# bird dog
# bird 0 0
# dog 0 2


print("CORRECTION:")

y_true = np.array(['a', 'b', 'c'])
y_hat = np.array(['a', 'b', 'c'])
print(f"{confusion_matrix_(y_true, y_hat) = }")
print("should return a numpy.array or pandas.DataFrame full of zeros except the diagonal which should be full of ones.")
print()

y_true = np.array(['a', 'b', 'c'])
y_hat = np.array(['c', 'a', 'b'])
print(f"{confusion_matrix_(y_hat, y_true) = }")
print(f"{confusion_matrix(y_hat, y_true) = }")
print('should return "np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])"')
print()

y_true = np.array(['a', 'a', 'a'])
y_hat = np.array(['a', 'a', 'a'])
print(f"{confusion_matrix_(y_true, y_hat) = }")
print("should return np.array([3])")
print()

y_true = np.array(['a', 'a', 'a'])
y_hat = np.array(['a', 'a', 'a'])
print(f"{confusion_matrix_(y_true, y_hat, labels=[]) = }")
print("return None, an empty np.array or an empty pandas.Dataframe.")
print()
