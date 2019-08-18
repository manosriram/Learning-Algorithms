import pandas as pd
import numpy as np
from numpy import ones

df = pd.read_csv("ex1data1.txt", names = ['X', 'y'])

y = df['y']
m = y.size

X = df.as_matrix(columns = ['X'])
X = np.append(ones((m,1)), X, axis=1)

theta = np.zeros(2)

def computeCost(X, y, theta):
	J = 0
	m = y.size
	h = np.dot(X, theta)
	sqrEr = np.sum(np.square(h - y))
	J = (sqrEr) / (2 * m)
	return J


cost = computeCost(X, y, theta)
print(cost)