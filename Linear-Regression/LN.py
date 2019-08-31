import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./ex1data1.txt", delimiter=',')

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]


admitted = df.loc[Y == 1]
not_admitted = df.loc[Y == 0]


plt.scatter(admitted.iloc[:, 0],
            admitted.iloc[:, 1], s=10, label='Admitted', color='blue')

plt.scatter(not_admitted.iloc[:, 0],
            not_admitted.iloc[:, 1], s=10, label='Not Admitted', color='red')

plt.show()
