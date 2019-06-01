import numpy as np
from collections import defaultdict


def get_probs(mask, state, action):
    """Get transition probabilities after taking action in a given state

    Args:
        state: The current state
        action: The action taken

    Returns:
        list of tuples: A list of tuples containing probability, next_state,
            reward and done

    """
    if action == 0:
        return [(1, state, 0, True)]

    result = []
    for roll in range(1, len(mask) + 1):
        if mask[roll - 1] == 1:
            leaf = (1.0 / len(mask), 1000, -state, True)
        else:
            leaf = (1.0 / len(mask), state + roll, roll, False)

        result.append(leaf)

    return result


def get_possible_states(mask, origin=[0]):
    result = []
    for node in origin:
        for i in range(1, len(mask) + 1):
            if (mask[i - 1] == 0) and ((node + i) not in result) and ((node + i) not in origin):
                result.append(node + i)

    return origin + result


def q_from_v(mask, V, s, gamma=1):
    q = np.zeros(2)
    for a in range(2):
        for prob, next_state, reward, done in get_probs(mask, s, a):
            q[a] += prob * (reward + gamma * V[next_state])
    return q


if __name__ == '__main__':
    #mask = [1, 1, 1, 0, 0, 0]
    #mask = [1,1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0]
    #mask = [1,1,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,0,0,1,0]
    mask = [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0]
    previous = [0]
    for i in range(4):
        states = get_possible_states(mask, origin=previous)
        previous = states

    states += [1000]
    print(states)

    gamma = 1
    theta = 1e-8
    V = defaultdict(lambda: 0) #np.zeros(len(states))
    while True:
        delta = 0
        for s in states:
            v = V[s]
            V[s] = max(q_from_v(mask, V, s, gamma))
            delta = max(delta,abs(V[s]-v))
        if delta < theta:
            break

    print(V[0])

