import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./dataSet2.txt")
x = df["cityPopulation"]
y = df["foodProfit"]

plt.plot(x, y, 'x')
plt.xlabel("Population of the City")
plt.ylabel("Profit of the Food Truck")
# plt.show()

iterations = 2000
alpha = 0.01
x = np.column_stack((np.ones(np.size(x)), x))

# 1/2m * Σithroughm : (htheta*x - y) ** 2


def costFunction(x, y, theta):
    m = y.size
    h = x.dot(theta)
    return 1 / 2 * m * (np.sum(np.square()))
