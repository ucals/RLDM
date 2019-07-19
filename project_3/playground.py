import numpy as np
from collections import defaultdict


def choice_experiments():
    x = np.random.randint(2, size=(5, 5))
    print(x)
    winner = np.argwhere(x == np.amax(x))
    print(winner)
    choice = np.random.choice(winner.shape[0])
    print(winner[choice])


def remove_negatives():
    x = [0, 1, 2, -1, 3, -2, 0]
    print(x)

    y = [a if a >= 0 else 0 for a in x]
    print(y)


if __name__ == '__main__':
    Q1 = defaultdict(lambda: np.zeros(5))
    print(Q1[0])
    Q1[0] = np.random.rand(5)
    print(Q1[0])
    print(np.argmax(Q1[0]))
