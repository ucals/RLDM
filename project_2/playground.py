import numpy as np
import matplotlib.pyplot as plt
import os

rank = np.arange(0, 17)
k = rank * 10 + 1
print(rank)
print(k)



x = np.linspace(0, 100, 100)
y = np.exp(-x / 1)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.grid()
plt.show()
