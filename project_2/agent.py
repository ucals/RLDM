import random
import sys
import numpy as np
import pandas as pd
from statistics import mean
from time import time
from datetime import timedelta
from collections import deque
from keras.layers import Input, Dense, Add, Lambda
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
import gym


class Agent(object):
    def __init__(self, gamma=0.99, alpha=0.0005, memory_capacity=10000,
                 batch_size=64, layers=[256, 256], dueling=True, double=True,
                 prioritized_er=False, tau=1.0, min_epsilon=0.05):
        self.env = gym.make('LunarLander-v2')
        self.gamma = gamma
        self.alpha = alpha
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size
        self.Q = self.build_model(layers=layers, dueling=dueling)
        self.Q_target = self.build_model(layers=layers, dueling=dueling)
        self.double = double
        #self.prioritized_er = prioritized_er
        self.tau = tau
        self.epsilon = 1.0
        self.min_epsilon = min_epsilon
        self.update_target_weights()

    def build_model(self, layers, dueling=True, plot=True):
        inputs = Input(shape=(self.env.observation_space.shape[0], ),
                       name='input')
        x = Dense(layers[0], activation='relu', name='dense_0')(inputs)
        for i in range(1, len(layers)):
            x = Dense(layers[i], activation='relu', name=f'dense_{i}')(x)

        if dueling:
            v = Dense(1, activation=None, name='v')(x)
            a = Dense(self.env.action_space.n, activation=None, name='a')(x)
            out = Lambda(lambda x: x[0] + (x[1] - K.mean(x[1])), name='q')([v, a])
            net_name = 'model_dueling'
        else:
            out = Dense(self.env.action_space.n, activation='linear', name='q')(x)
            net_name = 'model_simple'

        model = Model(inputs=inputs, outputs=out)
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

        if plot:
            plot_model(model, to_file=f'{net_name}.png', rankdir='LR')

        return model

    def act(self, state, explore=True):
        take_random_action = \
            np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon])
        if take_random_action and explore:
            return self.env.action_space.sample()
        else:
            s = np.reshape(state, (1, self.env.observation_space.shape[0]))
            q = self.Q.predict(s)[0]
            return np.argmax(q)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def update_target_weights(self):
        online_weights = self.Q.get_weights()
        target_weights = self.Q_target.get_weights()
        for i, ow in enumerate(online_weights):
            target_weights[i] = (1 - self.tau) * target_weights[i] + self.tau * ow

        self.Q_target.set_weights(target_weights)

    def save_model(self, filename='model'):
        model_json = self.Q.to_json()
        with open(f"{filename}.json", "w") as json_file:
            json_file.write(model_json)

        self.Q.save_weights(f"{filename}.h5")
        print(f'Model saved in "{filename}.json"", weights saved in '
              f'"{filename}.h5".')

    def load_model(self, filename='model'):
        json_file = open(f'{filename}.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.Q = model_from_json(loaded_model_json)
        self.Q_target = model_from_json(loaded_model_json)
        self.Q.load_weights(f"{filename}.h5")
        self.Q_target.load_weights(f"{filename}.h5")

    def experience_replay_vectorized(self):
        if len(self.memory) >= self.batch_size:
            selected_rows = np.random.choice(len(self.memory),
                                             size=self.batch_size,
                                             replace=False)
            mem = np.array(self.memory)
            s = np.vstack(mem[selected_rows, 0])
            a = mem[selected_rows, 1].astype(int)
            r = mem[selected_rows, 2]
            s_next = np.vstack(mem[selected_rows, 3])
            d = mem[selected_rows, 4].astype(int)

            y = self.Q_target.predict(s)
            q_t_next = self.Q_target.predict(s_next)

            if self.double:
                q_o_next = self.Q.predict(s_next)
                a_next = np.argmax(q_o_next, axis=1)
                y[np.arange(y.shape[0]), a] = r + self.gamma * \
                                              q_t_next[np.arange(y.shape[0]),
                                                       a_next] * (1 - d)
            else:
                y[np.arange(y.shape[0]), a] = r + self.gamma * \
                                              np.max(q_t_next, axis=1) * (1 - d)

            self.Q.fit(s, y, verbose=0)

    def experience_replay(self):
        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)

            s_next = np.vstack([item[3] for item in batch])
            q_o_next = self.Q.predict(s_next)
            a_next = [np.argmax(qn) for qn in q_o_next]

            q_t_next = self.Q_target.predict(s_next)

            s = np.vstack([item[0] for item in batch])
            q = self.Q_target.predict(s)

            for i, (state, action, reward, next_state, done) in enumerate(batch):
                if done:
                    new_q = reward
                else:
                    if self.double:
                        new_q = reward + self.gamma * q_t_next[i][a_next[i]]
                    else:
                        new_q = reward + self.gamma * np.max(q_t_next[i])

                q[i][action] = new_q

            self.Q.fit(s, q, batch_size=len(batch), verbose=0)

    def train(self, epsilon_decay, num_episodes=10000, runs_to_solve=100,
              max_t=1000, avg_solve_reward=200.0, freq_update_target=100,
              render=False, print_same_line=True, log_floydhub=False,
              stop_when_solved=True, score_filename='live_score.csv',
              vectorized=False):
        scores = deque(maxlen=runs_to_solve)
        df_scores = pd.DataFrame(columns=['episode', 'epsilon', 'score',
                                          'average', 'avg_q_values'])
        t_start = time()
        best_score = float('-inf')

        for i_episode in range(num_episodes):
            t = time()
            state = self.env.reset()
            points = 0
            states = [state]
            for i in range(max_t):
                if render:
                    self.env.render()

                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                points += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                states.append(state)
                if vectorized:
                    self.experience_replay_vectorized()
                else:
                    self.experience_replay()

                self.epsilon = epsilon_decay(self.epsilon, i_episode)
                if i % freq_update_target == 0:
                    self.update_target_weights()

                if done:
                    break

            episode_time = time() - t

            # Record avg state values
            q_values = self.Q.predict(np.array(states))

            # Check score
            if i_episode >= 0 and print_same_line:
                sys.stdout.write(b'\033[2A'.decode())

            scores.append(points)

            if mean(scores) > best_score:
                best_score = mean(scores)

            df_scores.loc[i_episode] = [int(i_episode), self.epsilon,
                                        int(points), mean(scores),
                                        np.mean(q_values)]
            df_scores.to_csv(score_filename, index=False)
            if not log_floydhub:
                print(f'Episode {i_episode}: {int(points)} score '
                      f'(epsilon: {self.epsilon:0.3f}, steps: {i + 1}, '
                      f'memory size: {len(self.memory)}, '
                      f'episode time: {episode_time:0.3f})')
                print(f'Past {len(scores)} runs: min {min(scores):0.0f}, '
                      f'max {max(scores):0.0f}, avg {mean(scores):0.1f}; '
                      f'Best score: {best_score:0.1f}')
            else:
                print(f'{{"metric": "average_score", "value": {mean(scores):0.3f}, '
                      f'"epoch": {int(i_episode)}}}')
                print(f'{{"metric": "score", "value": {points:0.3f}, '
                      f'"epoch": {int(i_episode)}}}')
                print(f'{{"metric": "epsilon", "value": {self.epsilon:0.3f}, '
                      f'"epoch": {int(i_episode)}}}')
                print(f'{{"metric": "average_q_value", '
                      f'"value": {np.mean(q_values):0.3f}, '
                      f'"epoch": {int(i_episode)}}}')
                print(f'{{"metric": "steps_until_done", '
                      f'"value": {i + 1}, "epoch": {int(i_episode)}}}')
                print(f'{{"metric": "episode_time", '
                      f'"value": {episode_time:0.3f}, "epoch": {int(i_episode)}}}')
                print(f'{{"metric": "memory_size", '
                      f'"value": {len(self.memory)}, "epoch": {int(i_episode)}}}')
                print(f'{{"metric": "best_score", '
                      f'"value": {best_score:0.3f}, "epoch": {int(i_episode)}}}')

            # Register first time agent solves
            if (mean(scores) >= avg_solve_reward) and stop_when_solved:
                print(f'\nSolved in {i_episode + 1} total episodes. Time to '
                      f'solve: {timedelta(seconds=time() - t_start)}')
                self.save_model()
                break

            # If training until maximum number of episodes, keep improving
            if (mean(scores) >= avg_solve_reward) and (mean(scores) >= best_score) \
                    and not stop_when_solved:
                best_score = mean(scores)
                best_score_episode = i_episode + 1
                print(f'\nNew best score found: {best_score:0.3f} in '
                      f'{best_score_episode} total episodes. Time since '
                      f'start: {timedelta(seconds=time() - t_start)}')
                self.save_model()
                print('\n\n')
