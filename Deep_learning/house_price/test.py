import numpy as np
a = np.array(
    [
        [1,2,3],
        [2,3,1],
        [6,8,5],
        [4,5,6]
    ]
)
x = a[:,:-1]
print(x[1])
print(a.shape)
print(np.random.randn(2,2))
# print(a.min(axis=0))