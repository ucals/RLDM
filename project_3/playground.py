import numpy as np


if __name__ == '__main__':
    x = np.random.randint(2, size=(5, 5))
    print(x)
    winner = np.argwhere(x == np.amax(x))
    print(winner)
    choice = np.random.choice(winner.shape[0])
    print(winner[choice])
