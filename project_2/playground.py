import numpy as np
import matplotlib.pyplot as plt
import os

#rank = np.arange(0, 17)
#k = rank * 10 + 1
#print(rank)
#print(k)


x = np.linspace(0, 500, 100)
y0 = np.exp(-x / 70)
y1 = np.exp(-x / 5)
y2 = np.exp(-x / 20)
y3 = np.exp(-x / 40)
y4 = np.exp(-x / 120)

y5 = 1 / (1 + np.exp(x/8 - 6))
y6 = 1 / (1 + np.exp(x/15 - 6))
y7 = 1 / (1 + np.exp(x/25 - 6))
y8 = 1 / (1 + np.exp(x/40 - 6))
y9 = 1 / (1 + np.exp(x/3 - 6))

fig, ax = plt.subplots()
ax.plot(x, y0, label='y0')
ax.plot(x, y1, label='y1')
ax.plot(x, y2, label='y2')
ax.plot(x, y3, label='y3')
ax.plot(x, y4, label='y4')

#ax.plot(x, y5)
#ax.plot(x, y6)
#ax.plot(x, y7)
#ax.plot(x, y8)
#ax.plot(x, y9)

ax.grid()
ax.legend()

plt.show()
