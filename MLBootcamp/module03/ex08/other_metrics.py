import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# • tp is the number of true positives,
# • fp is the number of false positives,
# • tn is the number of true negatives,
# • fn is the number of false negatives

def accuracy_score_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute the accuracy score.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    Return:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    # accuracy = (tp + tn) / (tp + fp + tn + fn)
    return (y_hat == y).mean()


def precision_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1) -> float:
    """
    Compute the precision score.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    # precision = tp / (tp + fp)
    tp = ((y_hat == y) & (y_hat == pos_label)).sum()
    fp = ((y_hat != y) & (y_hat == pos_label)).sum()
    return tp / (tp + fp)


def recall_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1) -> float:
    """
    Compute the recall score.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    # recall = tp / (tp + fn)
    tp = ((y_hat == y) & (y_hat == pos_label)).sum()
    fn = ((y_hat != y) & (y_hat != pos_label)).sum()
    return tp / (tp + fn)


def f1_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1) -> float:
    """
    Compute the f1 score.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    # F1score = (2*precision*recall) / (precision + recall)
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    return 2 * (precision * recall) / (precision + recall)


# Exemple 1
y_hat = np.array([[1], [1], [0], [1], [0], [0], [1], [1]])
y = np.array([[1], [0], [0], [1], [0], [1], [0], [0]])

# Accuracy
# your implementation
print(accuracy_score_(y, y_hat))
# Output:
0.5
# sklearn implementation
print(accuracy_score(y, y_hat))
# Output:
0.5
# # Precision
# your implementation
print(precision_score_(y, y_hat))
# Output:
0.4
# sklearn implementation
print(precision_score(y, y_hat))
# Output:
0.4
# # Recall
# your implementation
print(recall_score_(y, y_hat))
# Output:
0.6666666666666666
# sklearn implementation
print(recall_score(y, y_hat))
# Output:
0.6666666666666666
# F1-score
# your implementation
print(f1_score_(y, y_hat))
# Output:
0.5
# sklearn implementation
print(f1_score(y, y_hat))
# Output:
0.5


# Example 2:
y_hat = np.array(["norminet", "dog", "norminet",
                 "norminet", "dog", "dog", "dog", "dog"])
y = np.array(["dog", "dog", "norminet", "norminet",
             "dog", "norminet", "dog", "norminet"])
# Accuracy
# your implementation
print(accuracy_score_(y, y_hat))
# Output:
0.625
# sklearn implementation
print(accuracy_score(y, y_hat))
# Output:
0.625
# # Precision
# your implementation
print(precision_score_(y, y_hat, pos_label="dog"))
# Output:
0.6
# sklearn implementation
print(precision_score(y, y_hat, pos_label="dog"))
# Output:
0.6
# # Recall
# your implementation
print(recall_score_(y, y_hat, pos_label="dog"))
# Output:
0.75
# sklearn implementation
print(recall_score(y, y_hat, pos_label="dog"))
# Output:
0.75
# F1-score
# your implementation
print(f1_score_(y, y_hat, pos_label="dog"))
# Output:
0.6666666666666665
# sklearn implementation
print(f1_score(y, y_hat, pos_label="dog"))
# Output:
0.6666666666666665


# Example 3:
y_hat = np.array(["norminet", "dog", "norminet",
                 "norminet", "dog", "dog", "dog", "dog"])
y = np.array(["dog", "dog", "norminet", "norminet",
             "dog", "norminet", "dog", "norminet"])
# Accuracy
# your implementation
print(accuracy_score_(y, y_hat))
# Output:
0.625
# sklearn implementation
print(accuracy_score(y, y_hat))
# Output:
0.625
# # Precision
# # your implementation
print(precision_score_(y, y_hat, pos_label="norminet"))
# Output:
0.6666666666666666
# sklearn implementation
print(precision_score(y, y_hat, pos_label="norminet"))
# Output:
0.6666666666666666
# # Recall
# # your implementation
print(recall_score_(y, y_hat, pos_label="norminet"))
# Output:
0.5
# sklearn implementation
print(recall_score(y, y_hat, pos_label="norminet"))
# Output:
0.5
# F1-score
# your implementation
print(f1_score_(y, y_hat, pos_label="norminet"))
# Output:
0.5714285714285715
# sklearn implementation
print(f1_score(y, y_hat, pos_label="norminet"))
# Output:
0.5714285714285715
