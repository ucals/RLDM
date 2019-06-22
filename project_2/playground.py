import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 500, 100)
y = np.exp(-x / 10)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.grid()
plt.show()
