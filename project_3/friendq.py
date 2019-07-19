import numpy as np
import pandas as pd
from pulp import *
import environment
from collections import defaultdict


class FriendQ(object):
    def __init__(self, alpha0=0.1, alpha_min=0.001, alpha_decay=0.999995,
                 gamma=0.9):
        self.env = environment.Soccer()
        self.alpha0 = alpha0
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.Q0 = defaultdict(lambda: np.zeros((self.env.num_actions,
                                                self.env.num_actions)))
        self.Q1 = defaultdict(lambda: np.zeros((self.env.num_actions,
                                                self.env.num_actions)))

    def choose_actions(self, epsilon, state):
        take_random_action = np.random.choice([True, False], p=[epsilon,
                                                                1 - epsilon])
        if take_random_action:
            a0 = np.random.randint(self.env.num_actions)
        else:
            possibilities = np.argwhere(self.Q0[state] == np.amax(self.Q0[state]))
            choice = np.random.choice(possibilities.shape[0])
            a0 = possibilities[choice][0]

        take_random_action = np.random.choice([True, False], p=[epsilon,
                                                                1 - epsilon])
        if take_random_action:
            a1 = np.random.randint(self.env.num_actions)
        else:
            possibilities = np.argwhere(self.Q1[state] == np.amax(self.Q1[state]))
            choice = np.random.choice(possibilities.shape[0])
            a1 = possibilities[choice][0]

        return a0, a1

    def train(self, num_timesteps=1e6, epsilon_decay=0.999995,
              min_epsilon=0.001, print_same_line=True, log_filename='log.csv'):
        state_s = (0, 1), (0, 2), 0
        state_s_visits = 0
        state_s_actions = ()
        error = 0.
        count_errors = 0

        epsilon = 1.0
        t = 0
        i_episode = 0
        wins = [0, 0]

        alpha = self.alpha0

        df = pd.DataFrame(columns=['timestep', 'epsilon', 'alpha', 'error'])

        while t <= num_timesteps:
            i_episode += 1
            state = self.env.reset()
            while True:
                a0, a1 = self.choose_actions(epsilon, state)
                if (state == state_s) and (a0 == 0) and (a1 == 3):
                    state_s_visits += 1

                next_state, rewards, done, info = self.env.step(a0, a1)

                old_Q1 = self.Q1[state_s][3][0]
                v0 = np.max(self.Q0[next_state])
                v1 = np.max(self.Q1[next_state])

                if done:
                    self.Q0[state][a0][a1] = self.Q0[state][a0][a1] + alpha * (rewards[0] - self.Q0[state][a0][a1])
                    self.Q1[state][a1][a0] = self.Q1[state][a1][a0] + alpha * (rewards[1] - self.Q1[state][a1][a0])
                else:
                    self.Q0[state][a0][a1] = self.Q0[state][a0][a1] + alpha * (rewards[0] + self.gamma * v0 - self.Q0[state][a0][a1])
                    self.Q1[state][a1][a0] = self.Q1[state][a1][a0] + alpha * (rewards[1] + self.gamma * v1 - self.Q1[state][a1][a0])

                new_Q1 = self.Q1[state_s][3][0]

                if new_Q1 != old_Q1:
                    error = abs(new_Q1 - old_Q1)
                    df.loc[count_errors] = [int(t), epsilon, alpha, error]
                    df.to_csv(log_filename, index=False)
                    count_errors += 1

                # Compute statistics
                if rewards[0] == 100:
                    wins[0] += 1
                elif rewards[1] == 100:
                    wins[1] += 1

                # Log data in-screen
                if t > 0 and print_same_line:
                    sys.stdout.write(b'\033[3A'.decode())

                print(f'Episode {i_episode + 1:>4}, ε: {epsilon:0.3f}, α: {alpha:0.4f}, t: {t}, Wins: {wins}\n'
                      f'Actions: {(a0, a1)}, Rewards: {rewards}, Done: {done}, Info: {info}\n'
                      f'Error: {error:0.4f}, Visits in S state: {state_s_visits}, State S actions: {state_s_actions}')

                state = next_state
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                alpha = max(self.alpha_min, alpha * self.alpha_decay)
                #alpha = 1 / (t / self.alpha_min / num_timesteps + 1)

                t += 1

                if done:
                    break


if __name__ == '__main__':
    solver = FriendQ()
    solver.train(log_filename='log_friendq1.csv')

