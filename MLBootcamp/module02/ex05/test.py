import numpy as np
from mylinearregression import MyLinearRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLR([[1.], [1.], [1.], [1.], [1]])


print("predict_ 1\n", mylr.predict_(X), "\n")
print("loss elem 1 \n", mylr.loss_elem_(mylr.predict_(X), Y))
print("loss 1\n", mylr.loss_(mylr.predict_(X), Y), "\n")

mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print("my theta fit\n", mylr.thetas,  "\n")

print("predict_ 2\n", mylr.predict_(X), "\n")
print("loss elem 2\n", mylr.loss_elem_(mylr.predict_(X), Y))
print("loss 2\n", mylr.loss_(mylr.predict_(X), Y), "\n")
