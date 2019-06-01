import numpy as np
from collections import defaultdict
import sys
import environment as e


def generate_episode(env, policy):
    episode = []
    state = env.reset()
    while True:
        action = policy[state]
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def mc_prediction_q(env, num_episodes, policy, gamma=1.0):
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.num_actions))
    N = defaultdict(lambda: np.zeros(env.num_actions))
    Q = defaultdict(lambda: np.zeros(env.num_actions))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # generate an episode
        episode = generate_episode(env, policy)
        # obtain the states, actions, and rewards
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        # calculate and store the return for each visit in the episode
        for i, (state, action, reward) in enumerate(episode):
            returns_sum[state][action] += sum(rewards[i:]*discounts[:-(1+i)])
            N[state][action] += 1

    for state, values in returns_sum.items():
        for action in range(env.num_actions):
            Q[state][action] = values[action] / N[state][action]

    return Q


def generate_episode_from_Q(bj_env, Q, epsilon):
    policy = np.ones(env.num_actions) / env.num_actions
    episode = []
    state = bj_env.reset()
    while True:
        if state in Q:
            for action in range(env.num_actions):
                if action == np.argmax(Q[state]):
                    policy[action] = 1 - epsilon + epsilon / env.num_actions
                else:
                    policy[action] = epsilon / env.num_actions
        else:
            policy = np.ones(env.num_actions) / env.num_actions

        action = np.random.choice(np.arange(env.num_actions), p=policy)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def mc_control_alpha(env, num_episodes, alpha, gamma=1.0):
    nA = env.num_actions
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        lamb = 5 / num_episodes
        epsilon = np.exp(-lamb * i_episode)

        episode = generate_episode_from_Q(env, Q, epsilon)
        states, actions, rewards = zip(*episode)
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        for i, (state, action, reward) in enumerate(episode):
            Q[state][action] += (sum(rewards[i:]*discounts[:-(1+i)]) - Q[state][action]) * alpha

    policy = defaultdict(int)
    for state, values in Q.items():
        for action in range(env.num_actions):
            policy[state] = np.argmax(Q[state])

    return policy, Q


if __name__ == '__main__':
    test = True
    if test:
        env = e.Environment([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0,
                             0, 1, 0])
        policy_alpha, Q_alpha = mc_control_alpha(env, 2000000, 0.005)
        print(f'\n{env}: \n- Expected value training = {Q_alpha[0][1]}\n')
        Q = mc_prediction_q(env, 2000000, policy_alpha)
        print(f'\n- Expected value eval = {Q[0][1]}\n')


        #env = e.Environment([1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1,
        #                     0, 0, 1, 0])
        #policy_alpha, Q_alpha = mc_control_alpha(env, 1000000, 0.005)
        #print(f'\n{env}: \n- Expected value = {Q_alpha[0][1]}\n')

        #env = e.Environment([1, 1, 1, 0, 0, 0])
        #policy_alpha, Q_alpha = mc_control_alpha(env, 1000000, 0.005)
        #print(f'\n{env}: \n- Expected value = {Q_alpha[0][1]}\n')

    #x = [0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,0]
    #env = e.Environment(x)
    #policy_alpha, Q_alpha = mc_control_alpha(env, 1000000, 0.005)
    #print(f'\n{env}: \n- Expected value = {Q_alpha[0][1]}\n')

