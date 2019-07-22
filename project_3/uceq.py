import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import environment
import sys
from collections import defaultdict
import pulp


solvers.options['show_progress'] = False
solvers.options['msg_lev'] = 'GLP_MSG_OFF'
solvers.options['show_progress'] = False  # disable solver output
solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
solvers.options['LPX_K_MSGLEV'] = 0  # previous versions
np.set_printoptions(precision=3)


class uCEQ(object):
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
        self.P0 = defaultdict(lambda: np.ones(self.env.num_actions) /
                                      self.env.num_actions)
        self.P1 = defaultdict(lambda: np.ones(self.env.num_actions) /
                                      self.env.num_actions)

    def pulp_ce(self, r0, r1):
        num_vars = self.env.num_actions ** 2
        vars = pulp.LpVariable.dicts("Util", range(num_vars), 0, 1.0)

        # Create the 'prob' variable to contain the problem data
        prob = pulp.LpProblem("Chicken problem", pulp.LpMaximize)
        sum_rewards = r0.flatten() + r1.flatten()

        prob += pulp.lpSum([vars[i] * sum_rewards[i] for i in range(num_vars)])
        prob += pulp.lpSum([vars[i] for i in range(num_vars)]) == 1

        G = self.build_ce_constraints(r0, r1)
        for r in range(G.shape[0]):
            prob += pulp.lpSum([G[r, c] * vars[c] for c in range(G.shape[1])]) <= 0

        prob.solve()
        probabilities = np.array([vars[i].varValue for i in range(num_vars)])
        probabilities -= probabilities.min() + 0.
        probabilities = probabilities.reshape(self.env.num_actions, self.env.num_actions) / probabilities.sum(0)
        pi = np.sum(probabilities, axis=1)
        v = np.max(np.sum(probabilities * r0, axis=1))
        #v = np.sum(probabilities * r0)
        #v = sum([pi[a_] * r0[a_, a_opponent] for a_ in range(self.env.num_actions)])
        return pi, v

    def build_ce_constraints(self, r0, r1):
        num_vars = self.env.num_actions
        G = []
        # row player
        for i in range(num_vars): # action row i
            for j in range(num_vars): # action row j
                if i != j:
                    constraints = [0 for i in r0.flatten()]
                    base_idx = i * num_vars
                    comp_idx = j * num_vars
                    for k in range(num_vars):
                        constraints[base_idx+k] = (- r0.flatten()[base_idx+k]
                                                   + r0.flatten()[comp_idx+k])
                    G += [constraints]
        # col player
        for i in range(num_vars): # action column i
            for j in range(num_vars): # action column j
                if i != j:
                    constraints = [0 for i in r1.flatten()]
                    for k in range(num_vars):
                        constraints[i + (k * num_vars)] = (
                                - r1.flatten()[i + (k * num_vars)]
                                + r1.flatten()[j + (k * num_vars)])
                    G += [constraints]
        return np.matrix(G, dtype="float")

    def uce_old(self, r0, r1):
        na = r0.shape[0]
        nvars = na ** 2

        Q_flat = r0.flatten()
        opQ_flat = r1.flatten()

        # Minimize matrix c (*=-1 to maximize)
        c = -np.array(Q_flat + opQ_flat, dtype="float")
        c = matrix(c)

        # Inequality constraints G*x <= h
        G = np.empty((0, nvars))

        # Player constraints
        for i in range(na):  # action row i
            for j in range(na):  # action row j
                if i == j: continue
                constraint = np.zeros(nvars)
                base_idx = i * na
                comp_idx = j * na
                for _ in range(na):
                    constraint[base_idx + _] = Q_flat[comp_idx + _] - Q_flat[base_idx + _]

                G = np.vstack([G, constraint])

        # Opponent constraints
        Gopp = np.empty((0, nvars))
        for i in range(na):  # action row i
            for j in range(na):  # action row j
                if i == j: continue
                constraint = np.zeros(nvars)
                for _ in range(na):
                    constraint[i + _ * na] = opQ_flat[j + (_ * na)] - opQ_flat[i + (_ * na)]

                Gopp = np.vstack([Gopp, constraint])

        G = np.vstack([G, Gopp])
        G = np.matrix(G, dtype="float")
        G = np.vstack([G, -1. * np.eye(nvars)])
        h_size = len(G)
        G = matrix(G)
        h = np.array(np.zeros(h_size), dtype="float")
        h = matrix(h)

        # Equality constraints Ax = b
        A = np.matrix(np.ones(nvars), dtype="float")
        A = matrix(A)
        b = np.matrix(1, dtype="float")
        b = matrix(b)
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver='glpk')

        if sol['x'] is not None:
            probs = np.array(sol['x'].T)[0]
            # Scale and normalize to prevent negative probabilities
            probs -= probs.min() + 0.
            probs = probs.reshape((na, na)) / probs.sum(0)
            pi = np.sum(probs, axis=1)
            #v = sum([pi[a_] * r0[a_, a_opponent] for a_ in range(na)])
            v = np.sum(probs * r0)
            return pi, v
        else:
            return np.ones(5) / 5, 0

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

    def train(self, num_timesteps=1e6, epsilon_decay=0.999995,
              min_epsilon=0.001, print_same_line=True, log_filename='log.csv'):
        state_s = (0, 1), (0, 2), 0
        state_s_visits = 0
        error = 0.
        count_errors = 0

        epsilon = 1.0
        #epsilon_decay = 10 ** (np.log10(min_epsilon) / num_timesteps)

        t = 0
        i_episode = 0
        wins = [0, 0]

        alpha = self.alpha0
        #alpha_decay = 10 ** (np.log10(self.alpha_min) / num_timesteps)

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

                if done:
                    self.Q0[state][a0][a1] = (1 - alpha) * self.Q0[state][a0][a1] + alpha * rewards[0]
                    self.Q1[state][a1][a0] = (1 - alpha) * self.Q1[state][a1][a0] + alpha * rewards[1]
                else:
                    prob_actions0, v0 = self.pulp_ce(self.Q0[next_state], self.Q1[next_state])
                    #v0 = np.max(self.Q0[next_state])
                    self.P0[next_state] = prob_actions0

                    prob_actions1, v1 = self.pulp_ce(self.Q1[next_state], self.Q0[next_state])
                    #v1 = np.max(self.Q1[next_state])
                    self.P1[next_state] = prob_actions1

                    self.Q0[state][a0][a1] = (1 - alpha) * self.Q0[state][a0][a1] + alpha * (rewards[0] + self.gamma * v0)
                    self.Q1[state][a1][a0] = (1 - alpha) * self.Q1[state][a1][a0] + alpha * (rewards[1] + self.gamma * v1)

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
                      f'Error: {error:0.4f}, Visits in S state: {state_s_visits}, Q1: {new_Q1}')

                state = next_state
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                alpha = max(self.alpha_min, alpha * self.alpha_decay)

                #epsilon = max(min_epsilon, epsilon_decay ** t)
                #alpha = max(self.alpha_min, alpha * self.alpha_decay)
                #alpha = 1 / (t / self.alpha_min / num_timesteps + 1)
                #alpha = max(self.alpha_min, alpha_decay ** t)

                t += 1

                if done:
                    break


if __name__ == '__main__':
    solver = uCEQ()
    solver.train(log_filename='log_uceq4.csv')
