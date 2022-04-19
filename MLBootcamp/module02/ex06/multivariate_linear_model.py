import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR
data = pd.read_csv("../resources/spacecraft_data.csv")
# X = np.array(data[["Age"]])
# Y = np.array(data[["Sell_price"]])
# myLR_age = MyLR(theta=[[1000.0], [-1.0]], alpha=1e-3, max_iter=60000)
# myLR_age.fit_(X, Y)
# print(f"MSE Score: {myLR_age.mse_(X[:,0].reshape(-1,1),Y).sum()}")

X = np.array(data[["Age", "Thrust_power", "Terameters"]])
Y = np.array(data[["Sell_price"]])
my_lreg = MyLR(theta=[1.0, 1.0, 1.0, 1.0], alpha=1e-4, max_iter=100000)
# Example 0:
print(my_lreg.mse_(X, Y))

# Example 1:
# my_lreg.fit_(X, Y)
# print(my_lreg.theta)

# # Example 2:
# print(my_lreg.mse_(X, Y))
