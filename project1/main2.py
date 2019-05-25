import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from tqdm import tqdm
import random


class Environment(object):
    def __init__(self, size=5):
        self.s = size
        self.v_true = np.zeros(self.s)
        for i in range(self.s):
            self.v_true[i] = (i + 1) / (self.s + 1)

    def generate_data(self, num_sequences=10, num_training_sets=100):
        training_sets = []
        for i_training_set in range(num_training_sets):
            sequences = []
            for i_sequence in range(num_sequences):
                state = np.zeros(self.s)
                curr_index = int((self.s - 1) / 2)
                state[curr_index] = 1
                done = False
                sequence = [state.copy()]
                while not done:
                    next_index = np.random.choice([curr_index - 1, curr_index + 1])
                    if next_index < 0:
                        sequence.append(0)
                        break
                    elif next_index >= self.s:
                        sequence.append(1)
                        break
                    else:
                        state[curr_index] = 0
                        state[next_index] = 1
                        sequence.append(state.copy())
                        curr_index = next_index

                sequences.append(sequence)

            training_sets.append(sequences)

        return training_sets

    def print_data(self, training_sets):
        for s, sequences in enumerate(training_sets):
            print(f'TRAINING SET {s + 1}' + ('*' * 100))
            for i, sequence in enumerate(sequences):
                print(f'Sequence {i + 1}')
                for j, step in enumerate(sequence):
                    print(f'Step {j + 1}: {step}')

                print(' ')

    def estimate_v_tdlambda(self, lambda_, training_set, alpha=0.1, gamma=1.0):
        v = np.repeat(0.5, self.s)
        rms = np.zeros(len(training_set))

        for i, sequence in enumerate(training_set):
            state = sequence[0]
            curr_index = np.where(state == 1)
            #eligibility = np.zeros(self.s)
            for j, next_state in enumerate(sequence[1:]):
                if type(next_state) is np.ndarray:
                    reward = 0
                    next_index = np.where(next_state == 1)
                    v[curr_index] += alpha * (reward + gamma * v[next_index] - v[curr_index])
                    curr_index = next_index
                elif next_state == 1:
                    v[curr_index] += alpha * (reward - v[curr_index])
                elif next_state == 0:
                    v[curr_index] += alpha * (reward - v[curr_index])

            rms[i] = np.sqrt(np.mean((v - self.v_true)**2))

        return v, rms


        # v[0] = 0
        # v[self.env.size - 1] = 0
        # rms = np.zeros(num_episodes)
        #
        # for i_episode in range(num_episodes):
        #     state = self.env.reset()
        #     done = False
        #     eligibility = np.zeros(self.env.size)
        #     while not done:
        #         next_state, reward, done, info = self.env.step()
        #         eligibility *= lambda_ * gamma
        #         eligibility[state] += 1.0
        #
        #         td_error = reward + gamma * v[next_state] - v[state]
        #         v += + alpha * td_error * eligibility
        #
        #         state = next_state
        #
        #     error = self.v_true[1:-1] - v[1:-1]
        #     rms[i_episode] = np.sqrt(np.mean(error**2))
        #
        # return v[1:-1], rms


if __name__ == '__main__':

    env = Environment()
    training_sets = env.generate_data()
    ts = training_sets[0]
    v, rms = env.estimate_v_tdlambda(0, ts)
    print(v)
    print(rms[-1])

