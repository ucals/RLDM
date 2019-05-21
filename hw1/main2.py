import numpy as np
from collections import defaultdict
import sys
import environment as e


def policy_risk(state):
    return [0, 1]


def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = defaultdict(lambda: 0)
    i = 0
    while True:
        delta = 0
        i += 1
        print(i)
        for s in range(len(env.mask)):
            Vs = 0
            for a, action_prob in enumerate(policy(s)):
                for prob, next_state, reward, done in env.get_probs(s, a):
                    Vs += action_prob * prob * (reward + gamma * V[next_state])

            delta = max(delta, abs(Vs - V[s]))
            V[s] = Vs

        if delta < theta:
            break

    return V


def q_from_v(env, V, s, gamma=1):
    q = [0, 0]

    for a in range(2):
        for prob, next_state, reward, done in env.get_probs(s, a):
            q[a] += prob * (reward + gamma * V[next_state])
            if (s == 0) and (a == 0):
                print(prob, next_state, reward, done)
                print(q[a])

    return q


if __name__ == '__main__':
    env = e.Environment([1, 1, 1, 0, 0, 0])
    V = policy_evaluation(env, policy_risk, gamma=1, theta=1e-8)
    #Q = np.zeros([6, 2]) #defaultdict(lambda: [0, 0])
    #for s in range(len(env.mask)):
    #    Q[s] = q_from_v(env, V, s)

    #print(Q)

