import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4)
x = np.random.normal(3.0, 1.0, 1000)
y = np.random.normal(50.0, 10.0, 1000) / x
a = np.array(x)
b = np.array(y)

p9 = np.poly1d(np.polyfit(a, b, 9))
xp = np.linspace(0, 6, 300)
plt.ylim(0, 120)
plt.xlim(0, 7)
plt.scatter(x, y, s=15)
plt.plot(xp, p9(xp), color='red')
plt.show()
