import numpy as np
from mylinearregression import MyLinearRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLR([[1.], [1.], [1.], [1.], [1]])


print("predict 1\n\n", mylr.predict_(X), "\n\n")
print("loss elem 1\n\n", mylr.loss_elem_(X, Y), "\n\n")
print("loss 1\n\n", mylr.loss_(X, Y), "\n\n")

mylr.fit_(X, Y, alpha=1.6e-4, n_cycle=200000)
print(mylr.theta)

print(mylr.predict_(X))
print(mylr.loss_elem_(X, Y))
print(mylr.loss_(X, Y))
