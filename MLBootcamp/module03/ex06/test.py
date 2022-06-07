import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR

if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    mylr = MyLR([2, 0.5, 7.1, -4.3, 2.09])

    # Example 0:
    print(mylr.predict_(X))
    print()

    # Example 1:
    print(mylr.loss_(Y, (mylr.predict_(X))))
    print()

    # Example 2:
    mylr.fit_(X, Y)
    print(mylr.theta)
    print()

    # Example 3:
    print(mylr.predict_(X))
    print()

    # Example 4:
    print(mylr.loss_(Y, (mylr.predict_(X))))
    print()
