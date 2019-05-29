# -*- coding: utf-8 -*-
"""OMSCS Reinforcement Learning - CS-7642-O03 - Project #1 Solution

This code solves Project #1 from OMSCS Reinforcement Learning - CS-7642-O03,
generating the charts used in the report. Running it is straightforward:

    $ python final_solution.py

This will create the figure with all 3 charts in the file 'images/figure.png',
used in the report. It also creates them in individual files,
'images/figure1.png', 'images/figure2.png', and 'images/figure3.png', which are
replications of Sutton's Figure 3, 4 and 5 respectively.

Todo:
    * Write report
    * Write README.md
    * Test in ubuntu
    * Include a requirements.txt in the package
    * Create folder 'images' if it doesn't exist

Created by Carlos Souza (souza@gatech.edu). May-2019.

"""

import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from collections import defaultdict
from time import sleep
import argparse


class Environment(object):
    """Environment class.

    This class simulates the Random Walk environment. At initialization, it
    receives the random walk size. Then, at each time step, it updates the
    current state and returns the experience tuples with next state, reward, and
    whether it reached a terminal state. Inspired in OpenAI gym environments.

    Args:
        size (int): Size of the random walk, excluding terminals states. For
            example, in Sutton's original paper, random walk has size = 5.

    Attributes:
        size (int): Size of the random walk, including terminal states.
        current_state (int): Index representing the current state. For example:
            if the random walk has size 7 (including terminal states), in the
            beginning of the simulation, when in center, current_state is 3
            (middle point between 0 and 6). If at next time step the move is to
            the right, current_index become 4. If left, current_index become 2.
            In this example, current_state = 0 is the terminal state at extreme
            left (zero reward) and current_state = 6 is the terminal state at
            extreme right (reward +1).
        t (int): Number of time steps elapsed since beginning of simulation.

    """
    def __init__(self, size=5):
        self.size = size + 2  # Adding 2 terminal states at beginning and end
        self.current_state = None
        self.t = None
        self.reset()

    def reset(self):
        """Reset method.

        Resets the simulation. To be called at the beginning of each simulation.

        """
        self.current_state = int((self.size - 1) / 2)
        self.t = 0
        return self.current_state

    def step(self):
        """Step method.

        At each time step, this method rolls the dice, gets the next state, and
        updates the attributes, returing the experience tuple.

        Returns:
            Tuple with next state index, reward, done (True if in terminal
                state), and time step (counter since beginning of simulation).

        """
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
    """Solver class.

    This class uses the Environment class to generate the data used in the
    experiments, and perform the experiments 1 and 2 described in Sutton's
    original paper.

    Args:
        env (Environment): The environment used to generate data for the
            experiments. If None, it creates an environment.
        size (int): If no environment is provided, it creates an environment
            using this argument as the environment's size

    Attributes:
        env (Environment): The environment used to generate data for the
            experiments.
        v_true (np.array): Array with the true values of the state-value
            function, used to calculate the RMS error in the experiments.

    """
    def __init__(self, env=None, size=5):
        self.env = Environment(size=size) if env is None else env
        self.v_true = np.zeros(self.env.size)
        for i in range(self.env.size):
            self.v_true[i] = i / (self.env.size - 1)

    def generate_data(self, num_sets=100, num_episodes=10):
        """Generates the data used in the experiments.

        This function simulates random walks using the environment class, and
        returns the data for the experiments.

        Args:
            num_sets (int): Number of training sets to be generated
            num_episodes (int): Number of episodes (or sequences) to be
                generated in each training set.

        Returns:
            list: list of training sets. Each training set is a list of
                episodes. And each episode is a list of tuples containing state,
                reward, next_state, done (True if in terminal state), and info
                (extra information with time step, not used).

        """
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
        """Perform Experiment 1

        This function performs experiment 1 as described in Sutton's original
        article.

        Args:
            training_sets: Dataset created by `generate_data` function.
            lambda_: λ parameter in TD(λ) algorithm.
            alpha: α parameter, the learning rate (or step-size).
            gamma: γ parameter, the discount factor.
            theta: Θ parameter, the minimum threshold to stop the simulation.
            verbose: if True, prints information in the screen during simulation.
            show_tqdm: if True, shows progress bar.
            traces_mode: 'accumulating' or 'replacing', different types of
                eligibility traces update. In the original article (1988),
                Sutton used accumulating eligibility, the default option. Later,
                he developed the replacing eligibility idea.

        Returns:
            float: Root Mean Square (RMS) error between the state-value function
                estimate and its true value, averaged over the training sets.

        Raises:
            ValueError: If traces_mode not 'accumulating' or 'replacing'.

        """
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
                    eligibility = np.zeros(self.env.size)
                    for state, reward, next_state, done, info in episode:
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

                # Calculates difference between new and old state-value function
                delta = np.sqrt(np.mean((v[1:-1] - v_old[1:-1])**2))
                if verbose:
                    print(f'v_old: {v_old[1:-1]}; v: {v[1:-1]}; delta: {delta}')

                # Only updates V after all data is presented to learner
                v_old = v.copy()

            rms[i_set] = np.sqrt(np.mean((v[1:-1] - self.v_true[1:-1])**2))

        return np.mean(rms)

    def experiment_2(self, training_sets, lambda_, alpha=0.01, gamma=1.0,
                     traces_mode='accumulating'):
        """Perform Experiment 2

        This function performs experiment 2 as described in Sutton's original
        article.

        Args:
            training_sets: Dataset created by `generate_data` function.
            lambda_: λ parameter in TD(λ) algorithm.
            alpha: α parameter, the learning rate (or step-size)
            gamma: γ parameter, the discount factor
            traces_mode: 'accumulating' or 'replacing', different types of
                eligibility traces update. In the original article (1988),
                Sutton used accumulating eligibility, the default option. Later,
                he developed the replacing eligibility idea.

        Returns:
            float: Root Mean Square (RMS) error between the state-value function
                estimate and its true value, averaged over the training sets.

        Raises:
            ValueError: If traces_mode not 'accumulating' or 'replacing'.

        """
        rms = np.zeros(len(training_sets))
        for i_set, training_set in enumerate(training_sets):
            v = np.repeat(0.5, self.env.size)
            v_old = v.copy()

            for i_episode, episode in enumerate(training_set):
                eligibility = np.zeros(self.env.size)
                for state, reward, next_state, done, info in episode:
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

                # Updates state-value function at end of each episode
                v_old = v.copy()

            rms[i_set] = np.sqrt(np.mean((v[1:-1] - self.v_true[1:-1])**2))

        return np.mean(rms)

    def experiment_2_live(self, training_sets, file_v, file_e, file_error,
                          file_annotation, file_evol, lambdas=[0.0, 0.3],
                          alpha=0.2, gamma=1.0, traces_mode='accumulating',
                          show_tqdm=True):
        """Perform experiment 2 in parallel for 2 different λ values.

        This function performs experiment 2 as described in Sutton's original
        article, but for 2 different values of λ, in parallel. It is used in the
        function that creates the video animation. Check README.md for more
        information on how to use it.

        Args:
            training_sets: Dataset created by `generate_data` function.
            file_v: File path to save DataFrame with state-value estimates.
            file_e: File path to save DataFrame with eligibility values.
            file_error: File path to save DataFrame with RMS error values.
            file_annotation: File path to save text annotation.
            file_evol: File path to save DataFrame with RMS error evolution.
            lambdas: list of 2 λ parameters, to simulate 2 TD(λ) algorithms in
                parallel.
            alpha: α parameter, the learning rate (or step-size)
            gamma: γ parameter, the discount factor
            traces_mode: 'accumulating' or 'replacing', different types of
                eligibility traces update. In the original article (1988),
                Sutton used accumulating eligibility, the default option. Later,
                he developed the replacing eligibility idea.
            show_tqdm: if True, shows progress bar.

        Raises:
            ValueError: If traces_mode not 'accumulating' or 'replacing'.

        """
        def progress(f):
            if show_tqdm:
                return tqdm(f)
            else:
                return f

        state_letters = [chr(65 + i) for i in range(self.env.size - 2)]
        df_v = pd.DataFrame(index=pd.Index(['v_true'] + lambdas, name='lambda'),
                            columns=state_letters)
        df_e = pd.DataFrame(index=pd.Index(lambdas, name='eligibility'),
                            columns=state_letters)
        df_error = pd.DataFrame(index=pd.Index(range(1, len(training_sets[0]) + 1),
                                               name='episode'),
                                columns=lambdas)
        df_evol = pd.DataFrame(columns=lambdas)

        rms_1 = defaultdict(list)
        rms_2 = defaultdict(list)
        for i_set, training_set in enumerate(progress(training_sets)):
            sleep(3)
            v_1 = np.repeat(0.5, self.env.size)
            v_old_1 = v_1.copy()

            v_2 = np.repeat(0.5, self.env.size)
            v_old_2 = v_2.copy()

            for i_episode, episode in enumerate(training_set):
                state = episode[0][0]
                eligibility_1 = np.zeros(self.env.size)
                eligibility_2 = np.zeros(self.env.size)

                # Save dataframes to live plot
                df_v.iloc[0] = self.v_true[1:-1]
                df_v.iloc[1] = v_1[1:-1]
                df_v.iloc[2] = v_2[1:-1]
                df_v.to_csv(file_v)
                curr_state_str = state_letters[state - 1] if not episode[0][3] else 'terminal'
                annotation = f'Training set: {i_set}\nEpisode: ' \
                    f'{i_episode}\nTime step: {0}\nCurrent state: ' \
                    f'{curr_state_str}\nNext state: {curr_state_str}\nReward: {0}'
                with open(file_annotation, "w") as text_file:
                    text_file.write(annotation)

                for state, reward, next_state, done, info in episode:
                    sleep(.05)
                    if traces_mode == 'accumulating':
                        eligibility_1[state] += 1.0
                        eligibility_2[state] += 1.0
                    elif traces_mode == 'replacing':
                        eligibility_1[state] = 1.0
                        eligibility_2[state] = 1.0
                    else:
                        raise ValueError("'traces_mode' must be either "
                                         "'accumulating' or 'replacing'.")

                    # First lambda
                    td_target_1 = reward + gamma * v_old_1[next_state] if not done else reward
                    td_error_1 = td_target_1 - v_old_1[state]
                    v_1 += alpha * td_error_1 * eligibility_1

                    # Second lambda
                    td_target_2 = reward + gamma * v_old_2[next_state] if not done else reward
                    td_error_2 = td_target_2 - v_old_2[state]
                    v_2 += alpha * td_error_2 * eligibility_2

                    # Save dataframes to live plot
                    df_v.iloc[0] = self.v_true[1:-1]
                    df_v.iloc[1] = v_1[1:-1]
                    df_v.iloc[2] = v_2[1:-1]
                    df_v.to_csv(file_v)
                    df_e.iloc[0] = eligibility_1[1:-1]
                    df_e.iloc[1] = eligibility_2[1:-1]
                    df_e.to_csv(file_e)
                    curr_state_str = state_letters[state - 1]
                    next_state_str = state_letters[next_state - 1] if not done else 'terminal'
                    annotation = f'Training set: {i_set + 1}\nEpisode: ' \
                        f'{i_episode + 1}\nTime step: {info - 1}\nCurrent state: ' \
                        f'{curr_state_str}\nNext state: {next_state_str}\n' \
                        f'Reward: {reward}'
                    with open(file_annotation, "w") as text_file:
                        text_file.write(annotation)

                    eligibility_1 *= lambdas[0] * gamma
                    eligibility_2 *= lambdas[1] * gamma

                sleep(.2)

                v_old_1 = v_1.copy()
                v_old_2 = v_2.copy()

                e1 = np.sqrt(np.mean((v_1[1:-1] - self.v_true[1:-1])**2))
                e2 = np.sqrt(np.mean((v_2[1:-1] - self.v_true[1:-1])**2))
                df_evol.loc[len(df_evol.index)] = [e1, e2]
                df_evol.to_csv(file_evol)

                rms_1[i_episode].append(e1)
                rms_2[i_episode].append(e2)

            df_error[lambdas[0]] = [np.mean(rms) for k, rms in rms_1.items()]
            df_error[lambdas[1]] = [np.mean(rms) for k, rms in rms_2.items()]
            df_error.to_csv(file_error)


class Plot(object):
    """Plot class.

    This class uses the Solver class to perform the experiments 1 and 2 using
    different α (learning rates) and λ parameters, creating the charts shown in
    the report. It replicates Figures 3, 4 and 5 in Sutton's original article.

    Args:
        seed (int): If True, sets the seed. Useful to reproduce exact results.
        save_to_file (bool): If True, saves charts in files. Otherwise shows in
            screen.

    Attributes:
        solver (Solver): Solver object used in performing experiments.
        data (list): Dataset with all training sets used in the simulations.
        save_to_file (bool): If True, saves charts in files. Otherwise shows in
            screen.

    """
    def __init__(self, seed=None, save_to_file=True):
        self.solver = Solver()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.data = self.solver.generate_data()
        self.save_to_file = save_to_file

    def figure1(self, alpha=0.01, lambdas=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                axis=None):
        """Generate first chart

        This function generates our first chart, which replicates Sutton's
        Figure 3 in his original paper.

        Args:
            alpha (float): α parameter (learning rate) used in the experiments.
            lambdas (list): list of λ parameters used in the experiments.
            axis (matplotlib.axes.Axes): used to plot first chart in a combined
                single figure.

        """
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
            df.to_csv('images/figure1.csv')
            fig.savefig('images/figure1.png')
        else:
            plt.show()

    def figure2(self, alphas=np.linspace(0.0, 0.6, 13),
                lambdas=[0.0, 0.3, 0.8, 1.0], axis=None):
        """Generate second chart

        This function generates our second chart, which replicates Sutton's
        Figure 4 in his original paper.

        Args:
            alphas (list): list of α parameters (learning rates) used in the
                experiments.
            lambdas (list): list of λ parameters used in the experiments.
            axis (matplotlib.axes.Axes): used to plot second chart in a combined
                single figure.

        """
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
            df.to_csv('images/figure2.csv')
            fig.savefig('images/figure2.png')
        else:
            plt.show()

    def figure3(self, alphas=np.linspace(0.0, 0.6, 13),
                lambdas=np.linspace(0.0, 1.0, 11), axis=None):
        """Generate third chart

        This function generates our second chart, which replicates Sutton's
        Figure 5 in his original paper.

        Args:
            alphas (list): list of α parameters (learning rates) used in the
                experiments.
            lambdas (list): list of λ parameters used in the experiments.
            axis (matplotlib.axes.Axes): used to plot third chart in a combined
                single figure.

        """
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
            df.to_csv('images/figure3.csv')
            fig.savefig('images/figure3.png')
        else:
            plt.show()

    def live_plot(self):
        file_v = 'images/df_v.csv'
        file_e = 'images/df_e.csv'
        file_error = 'images/df_error.csv'
        file_evol = 'images/df_evol.csv'
        file_annotation = 'images/annotation.txt'
        self.solver.experiment_2_live(self.data, file_v, file_e, file_error,
                                      file_annotation, file_evol)

    def generate_all(self):
        """Generate all charts

        This function runs all experiments and generates all charts used in the
        report.

        """
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
    parser = argparse.ArgumentParser(description='Generate Project #1 figures.')
    parser.add_argument('-l', '--live', action="store_true", default=False,
                        help='Use this flag to generate DataFrames to create'
                             'video simulation. Check README.md to see how to'
                             'create the video simulation.')
    args = parser.parse_args()

    p = Plot(seed=200357)
    if not args.live:
        p.generate_all()
    else:
        p.live_plot()
