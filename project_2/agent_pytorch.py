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
                 prioritized_er=False, tau=1.0, min_epsilon=0.05, huber=False,
                 disable_cuda=False):
        if not disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

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

        self.huber = huber
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=self.alpha)
        self.update_target_weights()

    def build_model(self, layers, dueling=True):
        if dueling:
            return DuelingDQN(self.env.observation_space.shape[0], layers,
                              self.env.action_space.n).to(self.device)
        else:
            return DQN(self.env.observation_space.shape[0], layers,
                       self.env.action_space.n).to(self.device)

    def count_parameters(self):
        return sum(p.numel() for p in self.Q.parameters() if p.requires_grad)

    def act(self, state, explore=True):
        take_random_action = \
            np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon])
        if take_random_action and explore:
            return self.env.action_space.sample()
        else:
            st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            return torch.argmax(self.Q(st)).item()

    def remember(self, state, action, reward, next_state, done):
        if self.prioritized_er:
            st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            old_q = self.Q(st).detach().cpu().numpy()[0][action]
            st_next = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
            q_t_next = self.Q_target(st_next).detach().cpu().numpy()[0]
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
        if len(self.memory) >= self.batch_size:
            if self.prioritized_er:
                batch, idxs, is_weights = self.memory.sample(self.batch_size)
                is_weights = torch.from_numpy(is_weights).float().view((self.batch_size, 1)).to(self.device)
            else:
                batch = random.sample(self.memory, self.batch_size)

            batch_ = np.asarray(batch)
            s = np.stack(batch_[:, 0])
            a = np.stack(batch_[:, 1])
            r = np.stack(batch_[:, 2])
            s_p = np.stack(batch_[:, 3])
            d = np.stack(batch_[:, 4].astype(int))

            states = torch.from_numpy(s).to(self.device)
            y = self.Q(states).detach().cpu().numpy()
            next_states = torch.from_numpy(s_p).to(self.device)
            q_next_t = self.Q_target(next_states).detach().cpu().numpy()
            q_next_o = self.Q(next_states).detach().cpu().numpy()

            amax = np.argmax(q_next_o, axis=1)
            y[np.arange(y.shape[0]), a] = r + self.gamma * q_next_t[np.arange(y.shape[0]), amax] * (1 - d)

            # Perform a gradient descent step
            target = torch.from_numpy(y).to(self.device)
            old = self.Q(states).to(self.device)
            if self.prioritized_er:
                if self.huber:
                    t = torch.abs(target - old).to(self.device)
                    loss = (is_weights * torch.where(t < 1, 0.5 * t ** 2, t - 0.5)).mean().to(self.device)
                else:
                    loss = (is_weights * ((target - old) ** 2)).mean().to(self.device)
            else:
                if self.huber:
                    t = torch.abs(target - old).to(self.device)
                    loss = torch.where(t < 1, 0.5 * t ** 2, t - 0.5).to(self.device)
                else:
                    loss = ((target - old) ** 2).mean().to(self.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def experience_replay(self):
        if len(self.memory) >= self.batch_size:
            if self.prioritized_er:
                batch, idxs, is_weights = self.memory.sample(self.batch_size)
                is_weights = torch.from_numpy(is_weights).float().view((self.batch_size, 1)).to(self.device)
            else:
                batch = random.sample(self.memory, self.batch_size)

            states = torch.stack([torch.from_numpy(i[0]) for i in batch]).float().to(self.device)
            y = self.Q(states).detach().cpu().numpy()
            next_states = torch.stack([torch.from_numpy(i[3]) for i in batch]).float().to(self.device)
            q_t = self.Q_target(next_states).detach().cpu().numpy()

            # Do Q-learning update
            for i, (s, a, r, s_p, d) in enumerate(batch):
                if d:
                    y[i][a] = r
                else:
                    y[i][a] = r + self.gamma * np.max(q_t[i])

            # Perform a gradient descent step
            target = torch.from_numpy(y).to(self.device)
            old = self.Q(states).to(self.device)
            if self.prioritized_er:
                if self.huber:
                    t = torch.abs(target - old).to(self.device)
                    loss = (is_weights * torch.where(t < 1, 0.5 * t ** 2, t - 0.5)).mean().to(self.device)
                else:
                    loss = (is_weights * ((target - old) ** 2)).mean().to(self.device)
            else:
                if self.huber:
                    t = torch.abs(target - old).to(self.device)
                    loss = torch.where(t < 1, 0.5 * t ** 2, t - 0.5).to(self.device)
                else:
                    loss = ((target - old) ** 2).mean().to(self.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict_Q_values(self, list_of_states):
        states = torch.stack([torch.from_numpy(st) for st in list_of_states]).float().to(self.device)
        return self.Q(states).detach().cpu().numpy()

    def train(self, epsilon_decay, max_episodes=10000, runs_to_solve=100,
              max_t=1000, avg_solve_reward=200.0, freq_update_target=100,
              render=False, print_same_line=True, print_frequency=1, run_n='',
              log_floydhub=False, stop_when_solved=True, keep_learning=False,
              score_filename='live_score.csv', vectorized=False):
        scores = deque(maxlen=runs_to_solve)
        df_scores = pd.DataFrame(columns=['episode', 'epsilon', 'score',
                                          'average', 'avg_q_values'])
        t_start = time()
        best_score = float('-inf')
        solved = False

        for i_episode in range(max_episodes):
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
                if (not solved) or (solved and keep_learning):
                    self.remember(state, action, reward, next_state, done)

                state = next_state
                states.append(state)

                if (not solved) or (solved and keep_learning):
                    if vectorized:
                        self.experience_replay_vectorized()
                    else:
                        self.experience_replay()

                    self.epsilon = epsilon_decay(self.epsilon, i_episode)
                    if i % freq_update_target == 0:
                        self.update_target_weights()
                else:
                    self.epsilon = 0

                if done:
                    break

            episode_time = time() - t

            # Record score, state values and data from episode
            q_values = self.predict_Q_values(states)
            scores.append(points)
            if mean(scores) > best_score:
                best_score = mean(scores)

            df_scores.loc[i_episode] = [int(i_episode), self.epsilon,
                                        int(points), mean(scores),
                                        np.mean(q_values)]
            df_scores.to_csv(score_filename, index=False)

            # Log data in-screen
            if i_episode >= 0 and print_same_line:
                sys.stdout.write(b'\033[2A'.decode())

            if (i_episode + 1) % print_frequency == 0:
                self.log(i_episode, episode_time, scores, best_score, q_values,
                         i + 1, compact=not print_same_line, run_n=run_n,
                         log_floydhub=log_floydhub, t0=t_start)

            # Register first time agent solves
            if (mean(scores) >= avg_solve_reward) and stop_when_solved:
                if (i_episode + 1) % print_frequency != 0:
                    self.log(i_episode, episode_time, scores, best_score,
                             q_values, i + 1, compact=not print_same_line,
                             run_n=run_n, log_floydhub=log_floydhub, t0=t_start)

                print(f'{run_n} \tSolved in {i_episode + 1} total '
                      f'episodes. Training time: {timedelta(seconds=time() - t_start)}')
                self.save_model()
                break

            # If training until maximum number of episodes, keep improving
            if (mean(scores) >= avg_solve_reward) and (mean(scores) >= best_score) \
                    and not stop_when_solved:
                solved = True
                best_score = mean(scores)
                self.save_model()
                print(f'{run_n}\tEpisode {i_episode + 1:>3}: New best score\t'
                      f'past {len(scores):>3} runs avg: {best_score:0.1f}')

                if print_same_line:
                    print('\n\n')

        return df_scores

    def test(self, max_episodes=100, max_t=1000, render=False,
             print_same_line=True, print_frequency=1, run_n='',
             score_filename='live_score.csv'):
        scores = deque(maxlen=max_episodes)
        df_scores = pd.DataFrame(columns=['episode', 'epsilon', 'score',
                                          'average', 'avg_q_values'])
        t_start = time()
        best_score = float('-inf')
        self.epsilon = 0

        for i_episode in range(max_episodes):
            t = time()
            state = self.env.reset()
            points = 0
            states = [state]
            for i in range(max_t):
                if render:
                    self.env.render()

                action = self.act(state, explore=False)
                next_state, reward, done, info = self.env.step(action)
                points += reward
                state = next_state
                states.append(state)
                if done:
                    break

            episode_time = time() - t

            # Record score, state values and data from episode
            q_values = self.predict_Q_values(states)
            scores.append(points)
            if mean(scores) > best_score:
                best_score = mean(scores)

            df_scores.loc[i_episode] = [int(i_episode), self.epsilon,
                                        int(points), mean(scores),
                                        np.mean(q_values)]
            df_scores.to_csv(score_filename, index=False)

            # Log data in-screen
            if i_episode >= 0 and print_same_line:
                sys.stdout.write(b'\033[2A'.decode())

            if (i_episode + 1) % print_frequency == 0:
                self.log(i_episode, episode_time, scores, best_score, q_values,
                         i + 1, compact=not print_same_line, run_n=run_n,
                         t0=t_start)

        return df_scores

    def log(self, i_episode, episode_time, scores, best_score, q_values,
            steps_until_done, t0, compact=False, run_n='', log_floydhub=False):
        if not log_floydhub:
            if compact:
                print(f'{run_n} \tEpisode {i_episode + 1:>4}: '
                      f'{int(scores[-1]):>4} score\t'
                      f'<avg, min, max> past {len(scores):>3} runs: '
                      f'{mean(scores):0.1f}, {min(scores):0.0f}, {max(scores):0.0f}\t'
                      f'{timedelta(seconds=time() - t0)}')
            else:
                print(f'Episode {i_episode + 1}: {int(scores[-1])} score '
                      f'(epsilon: {self.epsilon:0.3f}, steps: {steps_until_done}, '
                      f'memory size: {len(self.memory)}, '
                      f'episode time: {episode_time:0.3f})')
                print(f'Past {len(scores)} runs: min {min(scores):0.0f}, '
                      f'max {max(scores):0.0f}, avg {mean(scores):0.1f}; '
                      f'Best score: {best_score:0.1f}')
        else:
            print(f'{{"metric": "average_score", "value": {mean(scores):0.3f}, '
                  f'"epoch": {int(i_episode)}}}')
            print(f'{{"metric": "score", "value": {scores[-1]:0.3f}, '
                  f'"epoch": {int(i_episode)}}}')
            print(f'{{"metric": "epsilon", "value": {self.epsilon:0.3f}, '
                  f'"epoch": {int(i_episode)}}}')
            print(f'{{"metric": "average_q_value", '
                  f'"value": {np.mean(q_values):0.3f}, '
                  f'"epoch": {int(i_episode)}}}')
            print(f'{{"metric": "steps_until_done", '
                  f'"value": {steps_until_done}, "epoch": {int(i_episode)}}}')
            print(f'{{"metric": "episode_time", '
                  f'"value": {episode_time:0.3f}, "epoch": {int(i_episode)}}}')
            print(f'{{"metric": "memory_size", '
                  f'"value": {len(self.memory)}, "epoch": {int(i_episode)}}}')
            print(f'{{"metric": "best_score", '
                  f'"value": {best_score:0.3f}, "epoch": {int(i_episode)}}}')