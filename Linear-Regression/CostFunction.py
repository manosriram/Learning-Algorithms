import pandas as pd
import numpy as np

data = pd.read_csv("./ex1data1.txt", sep=",", header=None)
m = len(data)
X = data[0]


def computeCost(X, y, theta):
    predicted = X * theta
    actual = y
    # sqrError = np.power((predicted - actual), 2)
    print(len(predicted))
    cost = 1 / (2 * m) * np.sum([1, 2, 3])
    return 1


X = [np.ones((97, 1), dtype="float"), data[1]]
y = data[1]

# cost = computeCost(X,y,theta)
