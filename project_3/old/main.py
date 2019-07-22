import numpy as np
import pandas as pd
from pulp import *
import environment
from collections import defaultdict


class MultiQ(object):
    def __init__(self, selection='Foe-Q', alpha0=0.1, alpha_min=0.001,
                 alpha_decay=0.999, gamma=0.9):
        self.env = environment.Soccer()
        if selection not in ['Friend-Q', 'Foe-Q', 'Q-learner', 'Correlated-Q']:
            raise ValueError("Selection must be either 'Friend-Q', 'Foe-Q', "
                             "'Q-learner' or 'Correlated-Q'")

        self.selection = selection
        self.alpha0 = alpha0
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.Q0 = defaultdict(lambda: np.zeros((self.env.num_actions,
                                                self.env.num_actions)))
        self.Q1 = defaultdict(lambda: np.zeros((self.env.num_actions,
                                                self.env.num_actions)))

    def minimax(self, r, maximin=False):
        if not maximin:
            prob = LpProblem("Foe-Q", LpMaximize)
        else:
            prob = LpProblem("Foe-Q", LpMinimize)

        v = LpVariable("TotalValue")
        p_0 = LpVariable("Action_0_Probability_Put", 0, 1)
        p_1 = LpVariable("Action_1_Probability_North", 0, 1)
        p_2 = LpVariable("Action_2_Probability_East", 0, 1)
        p_3 = LpVariable("Action_3_Probability_South", 0, 1)
        p_4 = LpVariable("Action_4_Probability_West", 0, 1)

        prob += v
        prob += p_0 + p_1 + p_2 + p_3 + p_4 == 1
        for c in range(r.shape[1]):
            if not maximin:
                prob += r[0][c] * p_0 + r[1][c] * p_1 + r[2][c] * p_2 + \
                        r[3][c] * p_3 + r[4][c] * p_4 >= v
            else:
                prob += r[0][c] * p_0 + r[1][c] * p_1 + r[2][c] * p_2 + \
                        r[3][c] * p_3 + r[4][c] * p_4 <= v

        prob.solve()
        return [p_0.varValue, p_1.varValue, p_2.varValue, p_3.varValue,
                p_4.varValue], v.varValue

    def choose_actions(self, epsilon, prob_actions0=None, prob_actions1=None):
        take_random_action = np.random.choice([True, False], p=[epsilon,
                                                                1 - epsilon])
        if take_random_action:
            a0 = np.random.randint(self.env.num_actions)
        else:
            a0 = np.random.choice(self.env.num_actions, p=prob_actions0)

        take_random_action = np.random.choice([True, False], p=[epsilon,
                                                                1 - epsilon])
        if take_random_action:
            a1 = np.random.randint(self.env.num_actions)
        else:
            a1 = np.random.choice(self.env.num_actions, p=prob_actions1)

        return a0, a1

    def train(self, num_timesteps=100000, render=False, epsilon_decay=1,
              min_epsilon=0.05, print_same_line=True, log_filename='log.csv'):
        state_s = (0, 1), (0, 2), 0
        state_s_visits = 0
        state_s_actions = ()
        error = 0.

        epsilon = 1.0
        t = 0
        i_episode = 0
        prob_actions0 = None
        prob_actions1 = None
        wins = [0, 0]

        alpha = self.alpha0

        df = pd.DataFrame(columns=['timestep', 'epsilon', 'alpha', 'error'])

        while True:
            i_episode += 1
            state = self.env.reset()
            done = False
            a0, a1 = self.choose_actions(epsilon, prob_actions0, prob_actions1)
            while not done:
                if render:
                    self.env.render()

                if state == state_s:
                    state_s_actions = (a0, a1)
                    if (a0 == 0) and (a1 == 3):
                        Q1_old = self.Q1[state][a0][a1]
                        state_s_visits += 1

                next_state, rewards, done, info = self.env.step(a0, a1)

                prob_actions0, v0 = self.minimax(self.Q0[next_state])
                #v0 = np.max(self.Q0[next_state])
                self.Q0[state][a0][a1] = (1 - alpha) * self.Q0[state][a0][a1] + alpha * ((1 - self.gamma) * rewards[0] + self.gamma * v0)

                prob_actions1, v1 = self.minimax(self.Q1[next_state], maximin=True)
                #v1 = np.max(self.Q1[next_state])
                self.Q1[state][a0][a1] = (1 - alpha) * self.Q1[state][a0][a1] + alpha * ((1 - self.gamma) * rewards[1] + self.gamma * v1)
                if (state == state_s) and (a0 == 0) and (a1 == 3):
                    error = abs(self.Q1[state][a0][a1] - Q1_old)
                    if error != 0:
                        df.loc[t] = [int(t), epsilon, alpha, error]
                        df.to_csv(log_filename, index=False)

                # Compute statistics
                if rewards[0] == 100:
                    wins[0] += 1
                elif rewards[1] == 100:
                    wins[1] += 1

                # Log data in-screen
                if t > 0 and print_same_line:
                    sys.stdout.write(b'\033[4A'.decode())

                print(f'Episode {i_episode + 1:>4}, ε: {epsilon:0.3f}, α: {alpha:0.4f}, t: {t}, Wins: {wins}\n'
                      f'Actions: {(a0, a1)}, Rewards: {rewards}, Done: {done}, Info: {info}\n'
                      f'Error: {error:0.4f}, Visits in S state: {state_s_visits}, State S actions: {state_s_actions}\n'
                      f'Prob actions 0: {prob_actions0}, Prob actions 1: {prob_actions1}')

                state = next_state
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                alpha = max(self.alpha_min, alpha * self.alpha_decay)
                a0, a1 = self.choose_actions(epsilon) #, prob_actions0, prob_actions1)
                # take_random_action = np.random.choice([True, False], p=[epsilon,
                #                                                         1 - epsilon])
                # if take_random_action:
                #     a0 = np.random.randint(self.env.num_actions)
                # else:
                #     a0 = np.unravel_index(self.Q0[state].argmax(), self.Q0[state].shape)[0]
                #
                # take_random_action = np.random.choice([True, False], p=[epsilon,
                #                                                         1 - epsilon])
                # if take_random_action:
                #     a1 = np.random.randint(self.env.num_actions)
                # else:
                #     a1 = np.unravel_index(self.Q0[state].argmax(), self.Q0[state].shape)[1]

                t += 1

                if done and render:
                    self.env.render()

            if t > num_timesteps:
                break

    def train_old(self, num_episodes=10000, render=False, epsilon_decay=0.999,
                  min_epsilon=0.05, print_same_line=True):
        state_s = (0, 1), (0, 2), 0
        state_s_visits = 0
        state_s_actions = ()
        error = 0.

        epsilon = 1.0
        t = 0
        prob_actions0 = None
        prob_actions1 = None
        wins = [0, 0]

        alpha = self.alpha0

        for i_episode in range(num_episodes):
            state = self.env.reset()
            done = False
            a0, a1 = self.choose_actions(epsilon, prob_actions0, prob_actions1)
            while not done:
                if render:
                    self.env.render()

                if state == state_s:
                    state_s_actions = (a0, a1)
                    if (a0 == 0) and (a1 == 3):
                        Q1_old = self.Q1[state][a0][a1]
                        state_s_visits += 1

                next_state, rewards, done, info = self.env.step(a0, a1)

                prob_actions0, v0 = self.minimax(self.Q0[next_state])
                self.Q0[state][a0][a1] = (1 - alpha) * self.Q0[state][a0][a1] + alpha * ((1 - self.gamma) * rewards[0] + self.gamma * v0)

                prob_actions1, v1 = self.minimax(self.Q1[next_state], maximin=True)
                self.Q1[state][a0][a1] = (1 - alpha) * self.Q1[state][a0][a1] + alpha * ((1 - self.gamma) * rewards[1] + self.gamma * v1)
                if (state == state_s) and (a0 == 0) and (a1 == 3):
                    error = abs(self.Q1[state][a0][a1] - Q1_old)

                # Compute statistics
                if rewards[0] == 100:
                    wins[0] += 1
                elif rewards[1] == 100:
                    wins[1] += 1

                # Log data in-screen
                if t > 0 and print_same_line:
                    sys.stdout.write(b'\033[4A'.decode())

                print(f'Episode {i_episode + 1:>4}, ε: {epsilon:0.3f}, α: {alpha:0.4f}, t: {t}, Wins: {wins}\n'
                      f'Actions: {(a0, a1)}, Rewards: {rewards}, Done: {done}, Info: {info}\n'
                      f'Error: {error:0.4f}, Visits in S state: {state_s_visits}, State S actions: {state_s_actions}\n'
                      f'Prob actions 0: {prob_actions0}, Prob actions 1: {prob_actions1}')

                state = next_state
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                alpha = max(self.alpha_min, alpha * self.alpha_decay)
                a0, a1 = self.choose_actions(epsilon, prob_actions0, prob_actions1)
                t += 1

                if done and render:
                    self.env.render()


if __name__ == '__main__':
    solver = MultiQ(selection='Friend-Q')
    solver.train()

