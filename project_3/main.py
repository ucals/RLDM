import numpy as np
from pulp import *
import environment
from collections import defaultdict


class MultiQ(object):
    def __init__(self, selection='Foe-Q', alpha=0.001, gamma=0.9):
        self.env = environment.Soccer()
        self.selection = selection
        self.alpha = alpha
        self.gamma = gamma
        self.Q1 = defaultdict(lambda: np.zeros((env.num_actions,
                                                env.num_actions)))
        self.Q2 = defaultdict(lambda: np.zeros((env.num_actions,
                                                env.num_actions)))

    def minimax(self, r):
        prob = LpProblem("Foe-Q", LpMaximize)

        v = LpVariable("TotalValue")
        p_0 = LpVariable("Action_0_Probability_Put", 0, 1)
        p_1 = LpVariable("Action_1_Probability_North", 0, 1)
        p_2 = LpVariable("Action_2_Probability_East", 0, 1)
        p_3 = LpVariable("Action_3_Probability_South", 0, 1)
        p_4 = LpVariable("Action_4_Probability_West", 0, 1)

        prob += v
        prob += p_0 + p_1 + p_2 + p_3 + p_4 == 1
        for c in range(r.shape[1]):
            prob += r[0][c] * p_0 + r[1][c] * p_1 + r[2][c] * p_2 + r[3][c] * \
                    p_3 + r[4][c] * p_4 >= v

        prob.solve()
        return [p_0.varValue, p_1.varValue, p_2.varValue, p_3.varValue,
                p_4.varValue], v.varValue

    def choose_actions(self, epsilon, prob_actions=None):
        # TODO select actions following ε-greedy policy
        a0 = np.random.randint(self.env.num_actions)
        a1 = np.random.randint(self.env.num_actions)
        return a0, a1

    def train(self, num_episodes=1000, render=False, epsilon_decay=0.99):
        epsilon = 1.0
        for i_episode in range(num_episodes):
            state = self.env.reset()
            done = False
            a0, a1 = self.choose_actions(state, epsilon)
            while not done:
                if render:
                    self.env.render()

                next_state, rewards, done, info = self.env.step(a0, a1)

                prob_actions, v = self.minimax(self.Q_matrix[next_state])
                self.Q_matrix[state][a0][a1] = (1 - self.alpha) * self.Q_matrix[state][a0][a1] + self.alpha * ((1 - self.gamma) * rewards[0] + self.gamma * v)

                state = next_state
                epsilon *= epsilon_decay
                a0, a1 = self.choose_actions(epsilon, prob_actions)

                if done and render:
                    env.render()


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
