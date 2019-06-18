import torch
import torch.nn as nn
import random
import sys
import numpy as np
import pandas as pd
from statistics import mean
from time import time
from datetime import timedelta
from collections import deque
import gym
from memory import Memory
from torch_networks import DQN, DuelingDQN


class Agent(object):
    def __init__(self, gamma=0.99, alpha=0.0005, memory_capacity=10000,
                 batch_size=64, layers=[512, 512], dueling=True, double=True,
                 prioritized_er=False, tau=1.0, min_epsilon=0.05):
        self.env = gym.make('LunarLander-v2')
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.Q = self.build_model(layers=layers, dueling=dueling)
        self.Q_target = self.build_model(layers=layers, dueling=dueling)
        self.double = double
        self.prioritized_er = prioritized_er
        if prioritized_er:
            self.memory = Memory(capacity=memory_capacity)
        else:
            self.memory = deque(maxlen=memory_capacity)

        self.tau = tau
        self.epsilon = 1.0
        self.min_epsilon = min_epsilon

        # self.loss_fn = nn.MSELoss()  # nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=self.alpha)
        self.update_target_weights()

    def build_model(self, layers, dueling=True, plot=True):
        if dueling:
            return DuelingDQN(self.env.observation_space.shape[0], layers[0],
                              self.env.action_space.n)
        else:
            return DQN(self.env.observation_space.shape[0], layers[0],
                       self.env.action_space.n)

    def count_parameters(self):
        return sum(p.numel() for p in self.Q.parameters() if p.requires_grad)

    def act(self, state, explore=True):
        take_random_action = \
            np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon])
        if take_random_action and explore:
            return self.env.action_space.sample()
        else:
            st = torch.from_numpy(state).float().unsqueeze(0)
            return torch.argmax(self.Q(st)).item()

    def remember(self, state, action, reward, next_state, done):
        if self.prioritized_er:
            st = torch.from_numpy(state).float().unsqueeze(0)
            old_q = self.Q(st).detach().numpy()[0][action]
            st_next = torch.from_numpy(next_state).float().unsqueeze(0)
            q_t_next = self.Q_target(st_next).detach().numpy()[0]
            if done:
                new_q = reward
            else:
                new_q = reward + self.gamma * np.max(q_t_next)

            error = abs(old_q - new_q)
            self.memory.add(error, (state, action, reward, next_state, done))
        else:
            self.memory.append([state, action, reward, next_state, done])

    def update_target_weights(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def save_model(self, filename='model.torch'):
        torch.save(self.Q.state_dict(), filename)

    def load_model(self, filename='model.torch'):
        self.Q.load_state_dict(torch.load(filename))
        self.Q.eval()

    def experience_replay_vectorized(self):
        # TODO update accordingly to experience_replay
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

            y = self.Q(torch.from_numpy(s).float()).detach().numpy()
            q_next_t = self.Q_target(torch.from_numpy(s_next).float()).detach().numpy()
            q_next_o = self.Q(torch.from_numpy(s_next).float()).detach().numpy()

            amax = np.argmax(q_next_o, axis=1)
            y[np.arange(y.shape[0]), a] = r + self.gamma * q_next_t[np.arange(y.shape[0]), amax] * (1 - d)

            # Perform a gradient descent step
            loss = self.loss_fn(torch.from_numpy(y), self.Q(torch.from_numpy(s).float()))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def experience_replay(self):
        if len(self.memory) >= self.batch_size:
            if self.prioritized_er:
                batch, idxs, is_weights = self.memory.sample(self.batch_size)
                is_weights = torch.from_numpy(is_weights).float().view((self.batch_size, 1))
            else:
                batch = random.sample(self.memory, self.batch_size)

            states = torch.stack([torch.from_numpy(i[0]) for i in batch]).float()
            y = self.Q(states).detach().numpy()
            next_states = torch.stack([torch.from_numpy(i[3]) for i in batch]).float()
            q_t = self.Q_target(next_states).detach().numpy()

            # Do Q-learning update
            for i, (s, a, r, s_p, d) in enumerate(batch):
                if d:
                    y[i][a] = r
                else:
                    y[i][a] = r + self.gamma * np.max(q_t[i])

            # Perform a gradient descent step
            # TODO implement Huber Loss:
            # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py (line 2178)
            if self.prioritized_er:
                loss = (is_weights * ((torch.from_numpy(y) - self.Q(states)) ** 2)).mean()
            else:
                loss = ((torch.from_numpy(y) - self.Q(states)) ** 2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict_Q_values(self, list_of_states):
        states = torch.stack([torch.from_numpy(st) for st in list_of_states]).float()
        return self.Q(states).detach().numpy()

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
            q_values = self.predict_Q_values(states)

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
