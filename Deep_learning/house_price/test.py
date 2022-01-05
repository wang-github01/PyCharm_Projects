import numpy as np
data = np.array([
    [1,2,3],
    [2,9,4],
    [3,8,5]
])
print(data.shape)
print(data[:,1])
print(data.max(axis=0))
print(data[:,1] - data.max(axis=0))