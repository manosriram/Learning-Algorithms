import pandas as pd
import numpy as np
from CostFunction import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
	J_H = []
	m = y.size
	for i in range(num_iters):
		x = X[1]
		h = np.dot(X, theta)
		theta = theta - (alpha / m) * np.dot(X.T, (h - y))

		J_H.append(computeCost(X, y, theta))
	
	
	return theta, J_H


df = pd.read_csv("ex1data1.txt", names = ['X', 'y'])
y = df['y']
m = y.size
X = df.as_matrix(columns=['X'])
X = np.append(np.ones((m,1)), X, axis=1)

theta = np.zeros(2)

print(gradientDescent(X, y, theta, 0.01, 1500)[0])