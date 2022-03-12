import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR


def draw_regression(x, y, MyLR):
    plt.plot(x, y, 'o', c='b')
    y_ = MyLR.predict_(x)
    plt.plot(x, y_, 'g--')
    plt.scatter(x, y_, c='g')
    plt.xlabel("Quantity of blue pills (in micrograms)")
    plt.ylabel("Space driving score")

    plt.show()


def draw_cost_function(x, y):
    plt.ylim((10, 50))
    plt.xlim((-13, -4.5))
    ran = 15
    upd = ran * 2 / 6
    for t0 in np.arange(89 - ran, 89 + ran, upd):
        cost_list = []
        theta_list = []
        for t1 in np.arange(-8 - 100, -8 + 100, 0.1):
            lr = MyLR(theta=[t0, t1], alpha=1e-3, max_iter=50000)
            y_ = lr.predict_(x)
            mse_c = lr.loss_(y, y_)
            cost_list.append(mse_c)
            theta_list.append(t1)
        label = "θ[0]=" + str(int(t0 * 10) / 10)
        plt.plot(theta_list, cost_list, label=label)
    plt.xlabel("Theta1")
    plt.ylabel("Cost function J(Theta0, Theta1)")
    plt.show()
    plt.cla()


data = pd.read_csv("../resources/are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
Yscore = np.array(data["Score"]).reshape(-1, 1)
# Example 1:
linear_model1 = MyLR(np.array([[89.0], [-8]]))
Y_model1 = linear_model1.predict_(Xpill)
print("MyLR =>", linear_model1.mse_(Yscore, Y_model1))
print("sklearn =>", mean_squared_error(Yscore, Y_model1))

# Example 2:
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model2 = linear_model2.predict_(Xpill)

print("MyLR =>", linear_model2.mse_(Yscore, Y_model2))

print("sklearn =>", mean_squared_error(Yscore, Y_model2))


draw_regression(Xpill, Yscore, linear_model1)
draw_cost_function(Xpill, Yscore)
