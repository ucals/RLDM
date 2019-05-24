import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque


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

    def estimate_v_td1(self, v0=None, num_episodes=100, gamma=1.0):
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


if __name__ == '__main__':
    s = Solver()
    rms_stack = None
    for i in range(100):
        v, rms = s.estimate_v_tdlambda(lambda_=0.15)
        if rms_stack is None:
            rms_stack = np.copy(rms)
        else:
            rms_stack = np.vstack((rms_stack, rms))

    rms_mean = np.mean(rms_stack, axis=0)
    plt.plot(rms_mean)
    plt.show()
