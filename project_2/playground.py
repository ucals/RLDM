import numpy as np


x = np.random.rand(3, 4)
print(x)

print(np.max(x))
print(np.max(x, axis=0))
print(np.max(x, axis=1))
