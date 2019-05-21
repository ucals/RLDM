import numpy as np


def get_possible_states(mask, origin=[0]):
    result = []
    for node in origin:
        for i in range(1, len(mask) + 1):
            if (mask[i - 1] == 0) and ((node + i) not in result) and ((node + i) not in origin):
                result.append(node + i)

    return origin + result


if __name__ == '__main__':
    mask = [1, 1, 1, 0, 0, 0]

    previous = [0]
    for i in range(10):
        states = get_possible_states(mask, origin=previous)
        previous = states

    print(states)
