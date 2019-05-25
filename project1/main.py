import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from tqdm import tqdm
import random


class Environment(object):
    def __init__(self, size=5):
        self.size = size + 2  # Adding 2 terminal states at beginning and end
        self.current_state = None
        self.t = None
        self.reset()

    def reset(self):
        self.current_state = int((self.size - 1) / 2)
        self.t = 0
        return self.current_state

    def step(self):
        self.t += 1
        next_state = np.random.choice([-1, 1]) + self.current_state
        self.current_state = next_state
        if next_state == self.size - 1:
            reward = 1
        else:
            reward = 0

        if next_state == self.size - 1 or next_state == 0:
            done = True
        else:
            done = False

        return next_state, reward, done, self.t


class Solver(object):
    def __init__(self, env=None, size=5):
        self.env = Environment(size=size) if env is None else env
        self.v_true = np.zeros(self.env.size)
        for i in range(self.env.size):
            self.v_true[i] = i / (self.env.size - 1)

    def estimate_v_td0(self, v0=None, num_episodes=100, alpha=0.05, gamma=1.0):
        v = np.repeat(0.5, self.env.size) if v0 is None else v0
        v[0] = 0
        v[self.env.size - 1] = 0
        rms = np.zeros(num_episodes)

        for i_episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                next_state, reward, done, info = self.env.step()
                v[state] += alpha * (reward + gamma * v[next_state] - v[state])
                state = next_state

            error = self.v_true[1:-1] - v[1:-1]
            rms[i_episode] = np.sqrt(np.mean(error**2))

        return v[1:-1], rms

    def estimate_v_mc(self, v0=None, num_episodes=100, gamma=1.0):
        v = np.repeat(0.5, self.env.size) if v0 is None else v0
        v[0] = 0
        v[self.env.size - 1] = 0
        rms = np.zeros(num_episodes)
        returns = defaultdict(list)

        for i_episode in range(num_episodes):
            state = self.env.reset()
            done = False
            experience = []
            while not done:
                next_state, reward, done, info = self.env.step()
                experience.append([state, reward])
                state = next_state

            states, rewards = zip(*experience)
            discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
            for i, state in enumerate(states):
                returns[state].append(sum(rewards[i:] * discounts[:-(i + 1)]))

            for k, x in returns.items():
                v[k] = np.mean(x)

            error = self.v_true[1:-1] - v[1:-1]
            rms[i_episode] = np.sqrt(np.mean(error**2))

        return v[1:-1], rms

    def estimate_v_nsteps(self, nsteps=2, v0=None, num_episodes=100, alpha=0.05,
                          gamma=1.0):
        v = np.repeat(0.5, self.env.size) if v0 is None else v0
        v[0] = 0
        v[self.env.size - 1] = 0
        rms = np.zeros(num_episodes)
        discounts = np.array([gamma ** i for i in range(nsteps + 1)])

        for i_episode in range(num_episodes):
            state = self.env.reset()
            done = False
            experience = deque(maxlen=nsteps)
            while not done:
                next_state, reward, done, info = self.env.step()
                experience.append([state, reward])
                if len(experience) == nsteps:
                    states, rewards = zip(*experience)
                    target = sum(rewards * discounts[:-1]) + discounts[-1] * v[next_state]
                    v[states[0]] += alpha * (target - v[states[0]])

                state = next_state

            error = self.v_true[1:-1] - v[1:-1]
            rms[i_episode] = np.sqrt(np.mean(error**2))

        return v[1:-1], rms

    def estimate_v_tdlambda(self, lambda_, v0=None, num_episodes=100,
                            alpha=0.05, gamma=1.0):
        v = np.repeat(0.5, self.env.size) if v0 is None else v0
        v[0] = 0
        v[self.env.size - 1] = 0
        rms = np.zeros(num_episodes)

        for i_episode in range(num_episodes):
            state = self.env.reset()
            done = False
            eligibility = np.zeros(self.env.size)
            while not done:
                next_state, reward, done, info = self.env.step()
                eligibility *= lambda_ * gamma
                eligibility[state] += 1.0

                td_error = reward + gamma * v[next_state] - v[state]
                v += + alpha * td_error * eligibility

                state = next_state

            error = self.v_true[1:-1] - v[1:-1]
            rms[i_episode] = np.sqrt(np.mean(error**2))

        return v[1:-1], rms

    def mean_estimate_tdlambda(self, runs, lambda_, v0=None, num_episodes=100,
                               alpha=0.05, gamma=1.0):
        rms_stack = None
        for i in range(runs):
            v, rms = self.estimate_v_tdlambda(lambda_=lambda_, v0=v0,
                                              num_episodes=num_episodes,
                                              alpha=alpha, gamma=gamma)
            if rms_stack is None:
                rms_stack = [np.copy(rms)]
            else:
                rms_stack = np.vstack((rms_stack, rms))

        return np.mean(rms_stack, axis=0)


class Plot(object):
    def __init__(self, seed=None):
        self.solver = Solver()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def figure3(self):
        alphas = [0.01] #np.linspace(0.0, 0.6, 13)
        lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        df = pd.DataFrame(index=lambdas, columns=alphas)
        for i_alpha, alpha in enumerate(tqdm(alphas)):
            for i_lambda, lambda_ in enumerate(lambdas):
                rms = self.solver.mean_estimate_tdlambda(runs=100,
                                                         lambda_=lambda_,
                                                         num_episodes=10,
                                                         alpha=alpha)
                df.iloc[i_lambda, i_alpha] = rms[-1] if rms[-1] < 1 else np.NaN

        df.to_csv('figure3.csv')
        print(df)
        plt.plot(df)
        plt.show()

    def figure4(self):
        alphas = np.linspace(0.0, 0.6, 13)
        lambdas = [0.0, 0.3, 0.8, 1.0]
        df = pd.DataFrame(index=alphas, columns=lambdas)

        for i_alpha, alpha in enumerate(tqdm(alphas)):
            for i_lambda, lambda_ in enumerate(lambdas):
                rms = self.solver.mean_estimate_tdlambda(runs=100,
                                                         lambda_=lambda_,
                                                         num_episodes=10,
                                                         alpha=alpha)
                df.iloc[i_alpha, i_lambda] = rms[-1] if rms[-1] < 1 else np.NaN

        # df['1.0'].loc[df.index > 0.4] = np.NaN
        df.to_csv('figure4.csv')
        print(df)
        plt.plot(df)
        plt.legend(df.columns.values)
        plt.show()

    def figure5(self):
        alphas = np.linspace(0.05, 0.2, 10) #np.linspace(0.0, 0.6, 13)
        lambdas = np.linspace(0.0, 1.0, 11)
        df = pd.DataFrame(index=lambdas, columns=alphas)
        for i_alpha, alpha in enumerate(tqdm(alphas)):
            for i_lambda, lambda_ in enumerate(lambdas):
                rms = self.solver.mean_estimate_tdlambda(runs=100,
                                                         lambda_=lambda_,
                                                         num_episodes=10,
                                                         alpha=alpha)
                df.iloc[i_lambda, i_alpha] = rms[-1] if rms[-1] < 1 else np.NaN

        df['best_alpha'] = df.min(axis=1)
        df.to_csv('figure5.csv')
        print(df)
        plt.plot(df['best_alpha']) #df.iloc[:, 4])
        plt.show()


if __name__ == '__main__':
    np.random.seed(10)
    random.seed(10)
    s = Solver()
    v, rms = s.estimate_v_tdlambda(lambda_=0.5, num_episodes=100, alpha=0.05, gamma=1.0)
    print(v)
    print(rms[-1])

    exit(0)
    p = Plot(seed=999) #999 #100
    p.figure3()

    exit(0)
    df = pd.read_csv('figure4.csv', index_col=0)
    print(df)
    print()

