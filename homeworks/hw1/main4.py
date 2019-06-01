import numpy as np
import pprint


# def get_possible_states(mask, experiences=[(0, 0, 0, 0)]):
#     result = []
#     for _, _, state, _ in experiences:
#         for i in range(1, len(mask) + 1):
#             if mask[i - 1] == 0:
#                 experience = (state, 1, state + i, i)
#                 if (experience not in result) and (experience not in experiences):
#                     result.append(experience)
#
#     return experiences + result
#
#
# if __name__ == '__main__':
#     pp = pprint.PrettyPrinter()
#     mask = [1, 1, 1, 0, 0, 0]
#
#     previous = [(0, 0, 0, 0)]
#     for i in range(4):
#         experiences = get_possible_states(mask, experiences=previous)
#         previous = experiences
#
#     pp.pprint(experiences)

def get_possible_states(mask, states=[(0, 0)]):
    result = []
    for bankroll, _ in states:
        for reward in range(1, len(mask) + 1):
            if mask[reward - 1] == 0:
                state = (bankroll + reward, reward)
                if (state not in result) and (state not in states):
                    result.append(state)

    return states + result


def get_T(mask, state):
    result = [((0, 0), sum(mask) / len(mask))]
    for reward in range(1, len(mask) + 1):
        if mask[reward - 1] == 0:
            state_prime = (state[0] + reward, reward)
            result.append((state_prime, 1.0 / len(mask)))

    return result


if __name__ == '__main__':
    pp = pprint.PrettyPrinter()
    mask = [1, 1, 1, 0, 0, 0]

    previous = [(0, 0)]
    for i in range(4):
        states = get_possible_states(mask, states=previous)
        previous = states

    pp.pprint(states)
    print(' ')
    pp.pprint(get_T(mask, state=(8, 4)))
    print(' ')

    gamma = 1
    V = np.zeros(len(states))
    Q = np.zeros((len(states), 2))
    policy = np.zeros(len(states), dtype=int)
    for iteration in range(100):
        for i_state, state in enumerate(states[:20]):
            Q[i_state][0] = 0 + gamma * 0
            Q[i_state][1] = state[1] + gamma * sum([prob * V[states.index(s_prime)] for s_prime, prob in get_T(mask, state)])
            policy[i_state] = int(np.argmax(Q[i_state]))
            print(policy[i_state])
            V[i_state] = Q[i_state][policy[i_state]]


    print(Q[0])

