import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


if __name__ == '__main__':

    #p = 0.24
    #v = [0.0, 0, 13.7, 0.5, -4.7, 17.6, 0.0]
    #r = [-1.2, -1.1, 4.8, 0, -1.7, 8.3, 2.2]

    p = 0.62
    v = [0.0, 0.0, 2.6, 19.5, 4.8, 0.6, 0.0]
    r = [-2.8, 4.4, -1.1, 1.1, -2.2, 2.7, 1.6]

    # Test
    #p = 0.5
    #v = [0, 3, 8, 2, 1, 2, 0]
    #r = [0, 0, 0, 4, 1, 1, 1]

    E = np.zeros(7)
    E[1] = p * (r[0] + v[1]) + (1 - p) * (r[1] + v[2])
    E[2] = p * (r[0] + r[2]) + (1 - p) * (r[1] + r[3]) + v[3]
    E[3] = p * (r[0] + r[2]) + (1 - p) * (r[1] + r[3]) + r[4] + v[4]
    E[4] = p * (r[0] + r[2]) + (1 - p) * (r[1] + r[3]) + r[4] + r[5] + v[5]
    E[5] = p * (r[0] + r[2]) + (1 - p) * (r[1] + r[3]) + r[4] + r[5] + r[6] + v[6]
    E[6] = p * (r[0] + r[2]) + (1 - p) * (r[1] + r[3]) + r[4] + r[5] + r[6] + 0 + 0
    print(f'E: {E}')

    def td(l):
        return E[1] * (1 - l) + E[2] * l * (1 - l) + E[3] * l**2 * (1 - l) + E[4] * l**3 * (1 - l) + E[5] * l**4 * (1 - l) + E[6] * l**5 * (1 - l) + l**6 * E[6]

    print(f'TD(0): {td(0)}')
    print(f'TD(1): {td(1)}')

    sol = fsolve(lambda l: td(l) - td(1), np.array([0.1]))[0]
    print(f'Lambda: {sol:0.6f}')
    print(f'Check: {td(sol)}')

    plot = False
    if plot:
        x = np.linspace(0, 1)
        plt.plot(x, td(x))
        plt.axhline(td(1), color='red')
        plt.show()

