import pandas as pd
import numpy as np

df = pd.read_csv("./ex1data1.txt", delimiter=',')

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]


def sigmoid(x):
    return 1 / 1 + np.exp(-x)


def net_input(theta, x):
    return np.dot(x, theta)


def probability(theta, y):
    return sigmoid(net_input(theta, y))


def costFunction(self, theta, X, y):
    m = X.shape[0]

    total_cost = -1 / m * \
        np.sum(y * np.log(probability(theta, X)) + (1 - y)
               * np.log(1 - probability(theta, x)))
    return total_cost


[m, n] = np.shape(X)

X = np.c_[np.ones((X.shape[0], 1)), X]
theta = np.zeros((X.shape[1], 1))
