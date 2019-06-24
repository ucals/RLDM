import numpy as np


def get_epsilon_decay(decay_function=0):
    def epsilon_decay_default(curr_epsilon, i_episode, min_epsilon=0.05):
        return max(min_epsilon, np.exp(-i_episode / 70))

    def epsilon_decay_1(curr_epsilon, i_episode, min_epsilon=0.05):
        return max(min_epsilon, np.exp(-i_episode / 5))

    def epsilon_decay_2(curr_epsilon, i_episode, min_epsilon=0.05):
        return max(min_epsilon, np.exp(-i_episode / 20))

    def epsilon_decay_3(curr_epsilon, i_episode, min_epsilon=0.05):
        return max(min_epsilon, np.exp(-i_episode / 40))

    def epsilon_decay_4(curr_epsilon, i_episode, min_epsilon=0.05):
        return max(min_epsilon, np.exp(-i_episode / 120))

    def epsilon_decay_5(curr_epsilon, i_episode, min_epsilon=0.05):
        y = 1 / (1 + np.exp(i_episode / 3 - 6))
        return max(min_epsilon, y)

    def epsilon_decay_6(curr_epsilon, i_episode, min_epsilon=0.05):
        y = 1 / (1 + np.exp(i_episode / 8 - 6))
        return max(min_epsilon, y)

    def epsilon_decay_7(curr_epsilon, i_episode, min_epsilon=0.05):
        y = 1 / (1 + np.exp(i_episode / 15 - 6))
        return max(min_epsilon, y)

    def epsilon_decay_8(curr_epsilon, i_episode, min_epsilon=0.05):
        y = 1 / (1 + np.exp(i_episode / 25 - 6))
        return max(min_epsilon, y)

    def epsilon_decay_9(curr_epsilon, i_episode, min_epsilon=0.05):
        y = 1 / (1 + np.exp(i_episode / 40 - 6))
        return max(min_epsilon, y)

    if decay_function == 0:
        return epsilon_decay_default
    elif decay_function == 1:
        return epsilon_decay_1
    elif decay_function == 2:
        return epsilon_decay_2
    elif decay_function == 3:
        return epsilon_decay_3
    elif decay_function == 4:
        return epsilon_decay_4
    elif decay_function == 5:
        return epsilon_decay_5
    elif decay_function == 6:
        return epsilon_decay_6
    elif decay_function == 7:
        return epsilon_decay_7
    elif decay_function == 8:
        return epsilon_decay_8
    elif decay_function == 9:
        return epsilon_decay_9
