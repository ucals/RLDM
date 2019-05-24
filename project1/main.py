import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    s = Solver()
    rms_stack = None
    for i in range(100):
        v, rms = s.estimate_v_td0()
        if rms_stack is None:
            rms_stack = np.copy(rms)
        else:
            rms_stack = np.vstack((rms_stack, rms))

    rms_mean = np.mean(rms_stack, axis=0)
    print(rms_mean)
    plt.plot(rms_mean)
    plt.show()
