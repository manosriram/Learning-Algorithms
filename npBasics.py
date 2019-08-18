import numpy as np

# 1-D Array
a = np.array([1,2,3])

# 2-D Array
b = np.array([(1,2,3), (4,5,6)])
c = np.array([(1.3,2.11,3.123), (4,5,6)], dtype=float)

# Initial Placeholders
z = np.zeros((3,4), dtype=int)
o = np.ones((3,4), dtype=int)
e = np.eye((3), dtype=int)
et = np.empty((3,5), dtype=int)
et = np.empty((3,5), dtype=int)
et = np.empty((3,5), dtype=int)
et = np.empty((3,5), dtype=int)


# Array Inspection
print(b.shape)
print(len(b))
print(b.ndim)
print(b.size)
print(b.dtype)
print(b.dtype.name)
b.astype(float)
print(b.dtype)