import numpy as np
import pandas as pd
from pulp import *
import environment
from collections import defaultdict
from abc import ABC, abstractmethod


class BaseLearner(ABC):
    def __init__(self, alpha0=0.1, alpha_min=0.001, alpha_decay=0.999995,
                 gamma=0.9, state_s=((0, 1), (0, 2), 0)):
        self.env = environment.Soccer()
        self.alpha0 = alpha0
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.state_s = state_s

    @abstractmethod
    def choose_actions(self, epsilon, state):
        pass

    @abstractmethod
    def update_Q_Pi(self, state, a0, a1, next_state, rewards, done, alpha):
        pass

    def train(self, num_timesteps=1e6, epsilon_decay=0.999995,
              min_epsilon=0.001, print_same_line=True, log_filename='log.csv'):
        state_s_visits = 0
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
                if (state == self.state_s) and (a1 == 3):
                    state_s_visits += 1

                next_state, rewards, done, info = self.env.step(a0, a1)

                old_Q1, new_Q1 = self.update_Q_Pi(state, a0, a1, next_state,
                                                  rewards, done, alpha)

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

                print(f'Episode {i_episode + 1:>4}, ε: {epsilon:0.3f}, '
                      f'α: {alpha:0.4f}, t: {t}, Wins: {wins}\n'
                      f'Actions: {(a0, a1)}, Rewards: {rewards}, Done: {done}, '
                      f'Info: {info}\n'
                      f'Error: {error:0.4f}, Visits in S: {state_s_visits}, '
                      f'Q1: {new_Q1}')

                state = next_state
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                alpha = max(self.alpha_min, alpha * self.alpha_decay)
                t += 1
                if done:
                    break


class QLearner(BaseLearner):
    def __init__(self, alpha0=0.1, alpha_min=0.001, alpha_decay=0.999995,
                 gamma=0.9, state_s=((0, 1), (0, 2), 0)):
        super(QLearner, self).__init__(alpha0, alpha_min, alpha_decay, gamma,
                                       state_s)
        self.Q0 = defaultdict(lambda: np.zeros(self.env.num_actions))
        self.Q1 = defaultdict(lambda: np.zeros(self.env.num_actions))

    def choose_actions(self, epsilon, state):
        take_random_action = np.random.choice([True, False], p=[epsilon,
                                                                1 - epsilon])
        if take_random_action:
            a0 = np.random.randint(self.env.num_actions)
        else:
            a0 = np.argmax(self.Q0[state])

        take_random_action = np.random.choice([True, False], p=[epsilon,
                                                                1 - epsilon])
        if take_random_action:
            a1 = np.random.randint(self.env.num_actions)
        else:
            a1 = np.argmax(self.Q1[state])

        return a0, a1

    def update_Q_Pi(self, state, a0, a1, next_state, rewards, done, alpha):
        old_Q1 = self.Q1[self.state_s][3]
        v0 = np.max(self.Q0[next_state])
        v1 = np.max(self.Q1[next_state])
        if done:
            self.Q0[state][a0] = self.Q0[state][a0] + \
                                 alpha * (rewards[0] - self.Q0[state][a0])
            self.Q1[state][a1] = self.Q1[state][a1] + \
                                 alpha * (rewards[1] - self.Q1[state][a1])
        else:
            self.Q0[state][a0] = self.Q0[state][a0] + alpha * \
                                 (rewards[0] + self.gamma * v0 -
                                  self.Q0[state][a0])
            self.Q1[state][a1] = self.Q1[state][a1] + alpha * \
                                 (rewards[1] + self.gamma * v1 -
                                  self.Q1[state][a1])

        new_Q1 = self.Q1[self.state_s][3]
        return old_Q1, new_Q1


class FriendQ(BaseLearner):
    def __init__(self, alpha0=0.1, alpha_min=0.001, alpha_decay=0.999995,
                 gamma=0.9, state_s=((0, 1), (0, 2), 0)):
        super(FriendQ, self).__init__(alpha0, alpha_min, alpha_decay, gamma,
                                       state_s)
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

    def update_Q_Pi(self, state, a0, a1, next_state, rewards, done, alpha):
        old_Q1 = self.Q1[self.state_s][3][0]
        v0 = np.max(self.Q0[next_state])
        v1 = np.max(self.Q1[next_state])
        if done:
            self.Q0[state][a0][a1] = self.Q0[state][a0][a1] + alpha * \
                                     (rewards[0] - self.Q0[state][a0][a1])
            self.Q1[state][a1][a0] = self.Q1[state][a1][a0] + alpha * \
                                     (rewards[1] - self.Q1[state][a1][a0])
        else:
            self.Q0[state][a0][a1] = self.Q0[state][a0][a1] + alpha * \
                                     (rewards[0] + self.gamma * v0 -
                                      self.Q0[state][a0][a1])
            self.Q1[state][a1][a0] = self.Q1[state][a1][a0] + alpha * \
                                     (rewards[1] + self.gamma * v1 -
                                      self.Q1[state][a1][a0])

        new_Q1 = self.Q1[self.state_s][3][0]
        return old_Q1, new_Q1


class FoeQ(BaseLearner):
    def __init__(self, alpha0=0.1, alpha_min=0.001, alpha_decay=0.999995,
                 gamma=0.9, state_s=((0, 1), (0, 2), 0)):
        super(FoeQ, self).__init__(alpha0, alpha_min, alpha_decay, gamma,
                                   state_s)
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
        probabilities = [p_0.varValue, p_1.varValue, p_2.varValue, p_3.varValue,
                         p_4.varValue]
        probabilities = [p if p >= 0 else 0 for p in probabilities]
        probabilities /= np.sum(probabilities)
        return probabilities, v.varValue

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

    def update_Q_Pi(self, state, a0, a1, next_state, rewards, done, alpha):
        old_Q1 = self.Q1[self.state_s][3][0]
        if done:
            self.Q0[state][a0][a1] = (1 - alpha) * self.Q0[state][a0][a1] + \
                                     alpha * rewards[0]
            self.Q1[state][a1][a0] = (1 - alpha) * self.Q1[state][a1][a0] + \
                                     alpha * rewards[1]
        else:
            prob_actions0, v0 = self.minimax(self.Q0[next_state])
            self.P0[next_state] = prob_actions0

            prob_actions1, v1 = self.minimax(self.Q1[next_state])
            self.P1[next_state] = prob_actions1

            self.Q0[state][a0][a1] = (1 - alpha) * self.Q0[state][a0][a1] + \
                                     alpha * (rewards[0] + self.gamma * v0)
            self.Q1[state][a1][a0] = (1 - alpha) * self.Q1[state][a1][a0] + \
                                     alpha * (rewards[1] + self.gamma * v1)

        new_Q1 = self.Q1[self.state_s][3][0]
        return old_Q1, new_Q1


class uCEQ(BaseLearner):
    def __init__(self, alpha0=0.1, alpha_min=0.001, alpha_decay=0.999995,
                 gamma=0.9, state_s=((0, 1), (0, 2), 0)):
        super(uCEQ, self).__init__(alpha0, alpha_min, alpha_decay, gamma,
                                   state_s)
        self.Q0 = defaultdict(lambda: np.zeros((self.env.num_actions,
                                                self.env.num_actions)))
        self.Q1 = defaultdict(lambda: np.zeros((self.env.num_actions,
                                                self.env.num_actions)))
        self.P0 = defaultdict(lambda: np.ones(self.env.num_actions) /
                                      self.env.num_actions)
        self.P1 = defaultdict(lambda: np.ones(self.env.num_actions) /
                                      self.env.num_actions)

    def pulp_ce(self, r0, r1):
        num_vars = self.env.num_actions ** 2
        vars = pulp.LpVariable.dicts("Probs", range(num_vars), 0, 1.0)

        prob = pulp.LpProblem("Soccer problem, uCE", pulp.LpMaximize)
        sum_rewards = r0.flatten() + r1.flatten()

        prob += pulp.lpSum([vars[i] * sum_rewards[i] for i in range(num_vars)])
        prob += pulp.lpSum([vars[i] for i in range(num_vars)]) == 1

        G = self.build_ce_constraints(r0, r1)
        for r in range(G.shape[0]):
            prob += pulp.lpSum([G[r, c] * vars[c] for c in
                                range(G.shape[1])]) <= 0

        prob.solve()
        probabilities = np.array([vars[i].varValue for i in range(num_vars)])
        probabilities -= probabilities.min() + 0.
        probabilities = probabilities.reshape(self.env.num_actions,
                                              self.env.num_actions) / \
            probabilities.sum(0)
        pi = np.sum(probabilities, axis=1)
        v = np.sum(probabilities * r0)
        return pi, v

    def build_ce_constraints(self, r0, r1):
        num_vars = self.env.num_actions
        G = []
        for i in range(num_vars):
            for j in range(num_vars):
                if i != j:
                    constraints = [0 for i in r0.flatten()]
                    base_idx = i * num_vars
                    comp_idx = j * num_vars
                    for k in range(num_vars):
                        constraints[base_idx+k] = (- r0.flatten()[base_idx+k]
                                                   + r0.flatten()[comp_idx+k])
                    G += [constraints]

        for i in range(num_vars):
            for j in range(num_vars):
                if i != j:
                    constraints = [0 for i in r1.flatten()]
                    for k in range(num_vars):
                        constraints[i + (k * num_vars)] = (
                                - r1.flatten()[i + (k * num_vars)]
                                + r1.flatten()[j + (k * num_vars)])
                    G += [constraints]

        return np.matrix(G, dtype="float")

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

    def update_Q_Pi(self, state, a0, a1, next_state, rewards, done, alpha):
        old_Q1 = self.Q1[self.state_s][3][0]
        if done:
            self.Q0[state][a0][a1] = (1 - alpha) * self.Q0[state][a0][a1] + \
                alpha * rewards[0]
            self.Q1[state][a1][a0] = (1 - alpha) * self.Q1[state][a1][a0] + \
                alpha * rewards[1]
        else:
            prob_actions0, v0 = self.pulp_ce(self.Q0[next_state],
                                             self.Q1[next_state])
            self.P0[next_state] = prob_actions0

            prob_actions1, v1 = self.pulp_ce(self.Q1[next_state],
                                             self.Q0[next_state])
            self.P1[next_state] = prob_actions1

            self.Q0[state][a0][a1] = (1 - alpha) * self.Q0[state][a0][a1] + \
                alpha * (rewards[0] + self.gamma * v0)
            self.Q1[state][a1][a0] = (1 - alpha) * self.Q1[state][a1][a0] + \
                alpha * (rewards[1] + self.gamma * v1)

        new_Q1 = self.Q1[self.state_s][3][0]
        return old_Q1, new_Q1
