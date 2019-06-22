import argparse
import gym
import os
from collections import deque
import numpy as np
from statistics import mean
from time import time
from datetime import timedelta
import torch
import torch.multiprocessing as mp
from torch_networks import DQN, DuelingDQN

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Asynchronous Methods for '
                                             'Deep Reinforcement Learning')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--episodes', type=int, default=1000, metavar='N',
                    help='number of episodes to train (default: 1000)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='gamma (default: 0.99)')
parser.add_argument('--update-target-interval', type=int, default=100, metavar='N',
                    help='how many steps to wait before updating target network')
parser.add_argument('--update-online-interval', type=int, default=10, metavar='N',
                    help='how many steps to wait before updating online network')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many episodes to wait before logging status')
parser.add_argument('--num-processes', type=int, default=16, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


def train(rank, args, Q, Q_target, device, env):
    name = mp.current_process().name
    torch.manual_seed(args.seed + rank)
    optimizer = torch.optim.RMSprop(Q.parameters(), lr=args.lr)
    epsilon = 1.0
    loss = torch.zeros(1)
    scores = deque(maxlen=100)
    t0 = time()

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        points = 0
        for i in range(1000):
            if np.random.choice([True, False], p=[epsilon, 1 - epsilon]):
                action = env.action_space.sample()
            else:
                st = torch.from_numpy(state).float().unsqueeze(0).to(device)
                action = torch.argmax(Q(st)).item()

            next_state, reward, done, info = env.step(action)
            points += reward

            st = torch.from_numpy(state).float().unsqueeze(0).to(device)
            old_q = Q(st).squeeze(0)[action]
            st_next = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
            q_t_next = Q_target(st_next).detach().cpu().numpy()[0]

            y = reward + args.gamma * np.max(q_t_next) if not done else reward
            y = torch.tensor(y).to(device)
            loss += (y - old_q) ** 2

            state = next_state
            epsilon = epsilon_decay(episode)

            if (rank == 0) and (i % args.update_target_interval == 0):
                update_target_network(Q, Q_target)

            if (i % args.update_online_interval == 0) or done:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = torch.zeros(1)

            if done:
                break

        scores.append(points)
        if episode % args.log_interval == 0:
            log(name, episode, scores, t0)


def update_target_network(Q, Q_target):
    Q_target.load_state_dict(Q.state_dict())


def epsilon_decay(i_episode, min_epsilon=0.05):
    return max(min_epsilon, np.exp(-i_episode / 70))


def log(name, episode, scores, t0):
    print(f'{name} \tEpisode {episode:>4}: '
          f'{int(scores[-1]):>4} score\t'
          f'<avg, min, max> past {len(scores):>3} runs: '
          f'{mean(scores):0.1f}, {min(scores):0.0f}, {max(scores):0.0f}\t'
          f'{timedelta(seconds=time() - t0)}')


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    mp.set_start_method('spawn')

    env = gym.make('LunarLander-v2')
    Q = DQN(env.observation_space.shape[0], [512, 512],
            env.action_space.n).to(device)
    Q.share_memory()
    Q_target = DQN(env.observation_space.shape[0], [512, 512],
                   env.action_space.n).to(device)

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, name=f'ag_{rank + 1:02d}',
                       args=(rank, args, Q, Q_target, device, env))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

