# TODO:
#   - Write report
#   - Write README.md
#   - Test in ubuntu
#   - Write comments

import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use('Agg')
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
            v = np.random.rand(self.env.size)
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

                        td_target = reward + gamma * v_old[next_state] if not done else reward
                        td_error = td_target - v_old[state]

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

                    td_target = reward + gamma * v_old[next_state] if not done else reward
                    td_error = td_target - v_old[state]
                    v += alpha * td_error * eligibility
                    eligibility *= lambda_ * gamma
                    state = next_state

                v_old = v.copy()

            rms[i_set] = np.sqrt(np.mean((v[1:-1] - self.v_true[1:-1])**2))

        return np.mean(rms)


class Plot(object):
    def __init__(self, seed=None, save_to_file=True):
        self.solver = Solver()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.data = self.solver.generate_data()
        self.save_to_file = save_to_file

    def figure1(self, alpha=0.01, lambdas=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                axis=None):
        lambdas = lambdas
        df = pd.DataFrame(index=lambdas, columns=['data'])
        for i_lambda, lambda_ in enumerate(tqdm(lambdas)):
            rms = self.solver.experiment_1(self.data, lambda_=lambda_,
                                           alpha=alpha)
            df.iloc[i_lambda, 0] = rms if rms < 1 else np.NaN

        fig, ax1 = plt.subplots()
        for ax in [ax1, axis]:
            ax.plot(df, 'o-', markersize=4)
            ax.set_ylabel('RMS Error', fontsize=14)
            ax.set_xlabel(r'$\lambda$', fontsize=18)
            ax.set_xticks(df.index.values)
            ax.annotate('Widrow-Hoff', xy=(df.iloc[-1].name, df.iloc[-1].values[0]),
                         xytext=(-105, -8), textcoords='offset pixels', fontsize=10)

        if self.save_to_file:
            df.to_csv('figure1.csv')
            fig.savefig('images/figure1.png')
        else:
            plt.show()

    def figure2(self, alphas=np.linspace(0.0, 0.6, 13),
                lambdas=[0.0, 0.3, 0.8, 1.0], axis=None):
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
        fig, ax1 = plt.subplots()
        for ax in [ax1, axis]:
            ax.plot(df, 'o-', markersize=4)
            ax.legend(df.columns.values, frameon=False)
            ax.set_ylabel('RMS Error', fontsize=14)
            ax.set_xlabel(r'$\alpha$', fontsize=18)
            legend = ax.get_legend()
            labels = [] if legend is None else [str(x._text) for x in legend.texts]
            handles = [] if legend is None else legend.legendHandles
            ax.legend(reversed(handles), reversed(labels), frameon=False)

        if self.save_to_file:
            df.to_csv('figure2.csv')
            fig.savefig('images/figure2.png')
        else:
            plt.show()

    def figure3(self, alphas=np.linspace(0.0, 0.6, 13),
                lambdas=np.linspace(0.0, 1.0, 11), axis=None):
        alphas = alphas
        lambdas = lambdas
        df = pd.DataFrame(index=lambdas, columns=alphas)
        for i_alpha, alpha in enumerate(tqdm(alphas)):
            for i_lambda, lambda_ in enumerate(lambdas):
                rms = self.solver.experiment_2(self.data, lambda_=lambda_,
                                               alpha=alpha)
                df.iloc[i_lambda, i_alpha] = rms

        df['best_alpha'] = df.min(axis=1)
        fig, ax1 = plt.subplots()
        for ax in [ax1, axis]:
            ax.plot(df['best_alpha'], 'o-', markersize=4)
            ax.set_ylabel('RMS Error using best α', fontsize=14)
            ax.set_xlabel(r'$\lambda$', fontsize=18) #
            ax.annotate('Widrow-Hoff', xy=(df.index[-1], df.iloc[-1, -1]),
                         xytext=(-105, -8), textcoords='offset pixels', fontsize=10)

        if self.save_to_file:
            df.to_csv('figure3.csv')
            fig.savefig('images/figure3.png')
        else:
            plt.show()

    def generate_all(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        plt.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.15,
                            wspace=0.27)
        print('Generating Figure 1...')
        self.figure1(axis=ax1)
        print('\nGenerating Figure 2...')
        self.figure2(axis=ax2)
        print('\nGenerating Figure 3...')
        self.figure3(axis=ax3)
        print('\nDONE!')
        if self.save_to_file:
            fig.savefig('images/figure.png')
        else:
            plt.show()


if __name__ == '__main__':
    p = Plot(seed=51180)
    p.generate_all()
