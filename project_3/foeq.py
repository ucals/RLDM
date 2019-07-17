import numpy as np
import pandas as pd
from pulp import *
import environment
from collections import defaultdict


class FoeQ(object):
    def __init__(self, alpha0=1.0, alpha_min=0.001, alpha_decay=0.999995,
                 gamma=0.9):
        self.env = environment.Soccer()
        self.alpha0 = alpha0
        self.alpha_min = alpha_min
        #self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.Q0 = defaultdict(lambda: np.zeros((self.env.num_actions,
                                                self.env.num_actions)))
        self.Q1 = defaultdict(lambda: np.zeros((self.env.num_actions,
                                                self.env.num_actions)))
        self.P0 = defaultdict(lambda: np.ones(self.env.num_actions) /
                                      self.env.num_actions)
        self.P1 = defaultdict(lambda: np.ones(self.env.num_actions) /
                                      self.env.num_actions)

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

    def choose_actions(self, epsilon, state):
        take_random_action = np.random.choice([True, False], p=[epsilon,
                                                                1 - epsilon])
        if take_random_action:
            a0 = np.random.randint(self.env.num_actions)
        else:
            a0 = np.random.choice(self.env.num_actions, p=self.P0[state])

        take_random_action = np.random.choice([True, False], p=[epsilon,
                                                                1 - epsilon])
        if take_random_action:
            a1 = np.random.randint(self.env.num_actions)
        else:
            a1 = np.random.choice(self.env.num_actions, p=self.P1[state])

        return a0, a1

    def train(self, num_timesteps=1e6, min_epsilon=0.001, print_same_line=True,
              log_filename='log.csv'):
        state_s = (0, 1), (0, 2), 0
        state_s_visits = 0
        state_s_actions = ()
        error = 0.
        count_errors = 0

        epsilon = 1.0
        epsilon_decay = 10 ** (np.log10(min_epsilon) / num_timesteps)
        t = 0
        i_episode = 0
        wins = [0, 0]

        alpha = self.alpha0
        alpha_decay = 10 ** (np.log10(self.alpha_min) / num_timesteps)

        df = pd.DataFrame(columns=['timestep', 'epsilon', 'alpha', 'error'])

        while t <= num_timesteps:
            i_episode += 1
            state = self.env.reset()
            while True:
                a0, a1 = self.choose_actions(epsilon, state)
                if (state == state_s) and (a0 == 0) and (a1 == 3):
                    state_s_visits += 1

                next_state, rewards, done, info = self.env.step(a0, a1)

                old_Q1 = self.Q1[state_s][0][3]

                if done:
                    self.Q0[state][a0][a1] = (1 - alpha) * self.Q0[state][a0][a1] + alpha * rewards[0]
                    self.Q1[state][a0][a1] = (1 - alpha) * self.Q1[state][a0][a1] + alpha * rewards[1]
                else:
                    prob_actions0, v0 = self.minimax(self.Q0[next_state])
                    self.P0[next_state] = [p if p >= 0 else 0 for p in prob_actions0]

                    # TODO understand whether Minimax or Maximin is the best appropriate way to calculate it
                    prob_actions1, v1 = self.minimax(self.Q1[next_state], maximin=True)
                    self.P1[next_state] = [p if p >= 0 else 0 for p in prob_actions1]

                    self.Q0[state][a0][a1] = (1 - alpha) * self.Q0[state][a0][a1] + alpha * (rewards[0] + self.gamma * v0)
                    self.Q1[state][a0][a1] = (1 - alpha) * self.Q1[state][a0][a1] + alpha * (rewards[1] + self.gamma * v1)

                new_Q1 = self.Q1[state_s][0][3]

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
                      f'Error: {error:0.4f}, Visits in S state: {state_s_visits}, old_Q1: {old_Q1}, new_Q1: {new_Q1}')

                state = next_state
                epsilon = max(min_epsilon, epsilon_decay ** t)
                #alpha = max(self.alpha_min, alpha * self.alpha_decay)
                #alpha = 1 / (t / self.alpha_min / num_timesteps + 1)
                alpha = max(self.alpha_min, alpha_decay ** t)

                t += 1

                if done:
                    break


if __name__ == '__main__':
    solver = FoeQ()
    solver.train()
