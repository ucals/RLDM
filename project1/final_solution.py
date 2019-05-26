# TODO:
#   - Write report
#   - Write README.md
#   - Test in ubuntu
#   - Write comments

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

    def generate_data(self, num_sets=100, num_episodes=10):
        sets = []
        for i_set in range(num_sets):
            episodes = []
            for i_episode in range(num_episodes):
                state = self.env.reset()
                done = False
                episode = []
                while not done:
                    next_state, reward, done, info = self.env.step()
                    episode.append([state, reward, next_state, done, info])
                    state = next_state

                episodes.append(episode)

            sets.append(episodes)

        return sets

    def experiment_1(self, training_sets, lambda_, alpha=0.01, gamma=1.0,
                     theta=1e-4, verbose=False, show_tqdm=False,
                     traces_mode='accumulating'):
        def progress(f):
            if show_tqdm:
                return tqdm(f)
            else:
                return f

        rms = np.zeros(len(training_sets))
        for i_set, training_set in enumerate(progress(training_sets)):
            v = np.repeat(0.5, self.env.size)
            v[0] = 0
            v[self.env.size - 1] = 0
            v_old = v.copy()
            delta = 1

            while delta > theta:
                for i_episode, episode in enumerate(training_set):
                    state = episode[0][0]
                    eligibility = np.zeros(self.env.size)
                    for _, reward, next_state, done, info in episode[1:]:
                        if traces_mode == 'accumulating':
                            eligibility[state] += 1.0
                        elif traces_mode == 'replacing':
                            eligibility[state] = 1.0
                        else:
                            raise ValueError("'traces_mode' must be either "
                                             "'accumulating' or 'replacing'.")

                        td_error = reward + gamma * v_old[next_state] - v_old[state]
                        v += alpha * td_error * eligibility
                        eligibility *= lambda_ * gamma
                        state = next_state

                delta = np.sqrt(np.mean((v[1:-1] - v_old[1:-1])**2))
                if verbose:
                    print(f'v_old: {v_old[1:-1]}; v: {v[1:-1]}; delta: {delta}')

                v_old = v.copy()

            rms[i_set] = np.sqrt(np.mean((v[1:-1] - self.v_true[1:-1])**2))

        return np.mean(rms)

    def experiment_2(self, training_sets, lambda_, alpha=0.01, gamma=1.0,
                     traces_mode='accumulating'):
        rms = np.zeros(len(training_sets))
        for i_set, training_set in enumerate(training_sets):
            v = np.repeat(0.5, self.env.size)
            v[0] = 0
            v[self.env.size - 1] = 0
            v_old = v.copy()

            for i_episode, episode in enumerate(training_set):
                state = episode[0][0]
                eligibility = np.zeros(self.env.size)
                for _, reward, next_state, done, info in episode[1:]:
                    if traces_mode == 'accumulating':
                        eligibility[state] += 1.0
                    elif traces_mode == 'replacing':
                        eligibility[state] = 1.0
                    else:
                        raise ValueError("'traces_mode' must be either "
                                         "'accumulating' or 'replacing'.")

                    td_error = reward + gamma * v_old[next_state] - v_old[state]
                    v += alpha * td_error * eligibility
                    eligibility *= lambda_ * gamma
                    state = next_state

                v_old = v.copy()

            rms[i_set] = np.sqrt(np.mean((v[1:-1] - self.v_true[1:-1])**2))

        return np.mean(rms)


class Plot(object):
    def __init__(self, seed=None):
        self.solver = Solver()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.data = self.solver.generate_data()

    def figure1(self, alpha=0.01, lambdas=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                save_to_file=True):
        lambdas = lambdas
        df = pd.DataFrame(index=lambdas, columns=['data'])
        for i_lambda, lambda_ in enumerate(tqdm(lambdas)):
            rms = self.solver.experiment_1(self.data, lambda_=lambda_,
                                           alpha=alpha)
            df.iloc[i_lambda, 0] = rms if rms < 1 else np.NaN

        fig = plt.figure()
        plt.plot(df, 'o-', markersize=4)
        plt.ylabel('RMS Error', fontsize=14)
        plt.xlabel('λ', fontsize=18)
        plt.xticks(df.index.values)
        plt.annotate('Widrow-Hoff', xy=(df.iloc[-1].name, df.iloc[-1].values[0]),
                     xytext=(-135, -8), textcoords='offset pixels', fontsize=14)
        if save_to_file:
            df.to_csv('figure1.csv')
            fig.savefig('figure1.png')
        else:
            plt.show()

    def figure2(self, alphas=np.linspace(0.0, 0.6, 13),
                lambdas=[0.0, 0.3, 0.8, 1.0], save_to_file=True):
        alphas = alphas
        lambdas = lambdas
        df = pd.DataFrame(index=alphas, columns=lambdas)
        for i_alpha, alpha in enumerate(tqdm(alphas)):
            for i_lambda, lambda_ in enumerate(lambdas):
                rms = self.solver.experiment_2(self.data, lambda_=lambda_,
                                               alpha=alpha)
                df.iloc[i_alpha, i_lambda] = rms if rms < 1 else np.NaN

        df.columns = [f'λ = {a}' for a in df.columns]
        df.rename(columns={'λ = 1.0': 'λ = 1.0 (Widrow-Hoff)'}, inplace=True)
        fig = plt.figure()
        plt.plot(df, 'o-', markersize=4)
        plt.legend(df.columns.values, frameon=False)
        plt.ylabel('RMS Error', fontsize=14)
        plt.xlabel('α', fontsize=18)
        if save_to_file:
            df.to_csv('figure2.csv')
            fig.savefig('figure2.png')
        else:
            plt.show()

    def figure3(self, alphas=np.linspace(0.0, 0.6, 13),
                lambdas=np.linspace(0.0, 1.0, 11), save_to_file=True):
        alphas = alphas
        lambdas = lambdas
        df = pd.DataFrame(index=lambdas, columns=alphas)
        for i_alpha, alpha in enumerate(tqdm(alphas)):
            for i_lambda, lambda_ in enumerate(lambdas):
                rms = self.solver.experiment_2(self.data, lambda_=lambda_,
                                               alpha=alpha)
                df.iloc[i_lambda, i_alpha] = rms

        df['best_alpha'] = df.min(axis=1)
        fig = plt.figure()
        plt.plot(df['best_alpha'], 'o-', markersize=4)
        plt.ylabel('RMS Error using best α', fontsize=14)
        plt.xlabel('λ', fontsize=18)
        plt.annotate('Widrow-Hoff', xy=(df.index[-1], df.iloc[-1, -1]),
                     xytext=(-135, -8), textcoords='offset pixels', fontsize=14)
        if save_to_file:
            df.to_csv('figure3.csv')
            fig.savefig('figure3.png')
        else:
            plt.show()


if __name__ == '__main__':
    p = Plot(seed=51180)  # 121033
    print('Generating Figure 1...')
    p.figure1()
    print('\nGenerating Figure 2...')
    p.figure2()
    print('\nGenerating Figure 3...')
    p.figure3()
    print('\nDONE!')
