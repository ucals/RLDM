import numpy as np
import environment
from collections import defaultdict


class MultiQ(object):
    def __init__(self, selection='Foe-Q', alpha=0.001, gamma=0.9):
        self.env = environment.Soccer()
        self.selection = selection
        self.alpha = alpha
        self.gamma = gamma

    def train(self):
        pass


if __name__ == '__main__':
    env = environment.Soccer()
    observation = env.reset()

    env._pos_p0 = [1, 1]
    env._pos_p1 = [1, 2]
    env._ball = 1
    env.render()

    observation, rewards, done, info = env.step(3, 4)
    print(f'Actions: {[3, 4]}, Rewards: {rewards}, Done: {done}, Info: {info}')

    env.render()

    Q_matrix = defaultdict(lambda: np.zeros((env.num_actions, env.num_actions)))
    s_0 = ((0, 1), (1, 2), 0)
    s_1 = ((1, 1), (0, 0), 1)

    Q_matrix[s_0][1][2] = 1
    print(Q_matrix[s_0])
    print(Q_matrix[s_1])
    Q_matrix[s_0][1][2] += 1
    print(Q_matrix[s_0])
