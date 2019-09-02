import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../Datasets/bostonHousing.csv")
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data.columns = housing_colnames


def predictPrice(x, theta):
    return np.dot(x, theta)


def plotData(x, theta, y):
    yV = predictPrice(x, theta)
    plt.xlim(0, 20)
    plt.ylim(-10, 60)
    plt.xlabel('No. of Rooms in the house')
    plt.ylabel('Price of house')
    plt.plot(x, y, '.', x, yV, '-')
    plt.show()


def calculateCost(x, theta, y):
    prediction = predictPrice(x, theta)
    return ((prediction - y) ** 2).mean() / 2


predictor = data["RM"]
x = np.column_stack((np.ones(len(predictor)), predictor))
y = data["MEDV"]
alpha = 0.01
num_iters = 5000

theta0 = []
theta1 = []
costs = []
theta = np.zeros(2)

for i in range(num_iters):
    pred = predictPrice(x, theta)
    t0 = theta[0] - alpha * (pred - y).mean()
    t1 = theta[1] - alpha * ((pred - y) * x[:, 1]).mean()
    theta = np.array([t0, t1])
    J = calculateCost(x, theta, y)
    theta0.append(t0)
    theta1.append(t1)
    costs.append(J)

    if i % 500 == 0:
        plotData(x, theta, y)

    print("Iteration : {}\ntheta : {}\ncost : {}".format(i, theta, J))
