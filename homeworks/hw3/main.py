import numpy as np
import json
import tester


def generate_mdp(num_states=3, num_actions=2, num_transitions=2, gamma=0.75):
    #l = np.arange(num_states)
    #print(l)
    #c = np.random.choice(l, 2)
    #print(c)
    #d = np.setdiff1d(l, c)
    #print(d)

    mdp = dict()
    mdp['gamma'] = gamma
    mdp['states'] = list()

    for s in range(num_states):
        state = dict()
        state['id'] = s
        state['actions'] = list()
        for a in range(num_actions):
            action = dict()
            action['id'] = a
            action['transitions'] = list()
            level = binary_level(s)
            possible_states = get_next_possible_states(level) # np.arange(num_states)
            for t in range(num_transitions):
                transition = dict()
                transition['id'] = t
                transition['probability'] = 1.0 / num_transitions
                #transition['reward'] = np.random.randint(-1, 2)
                if s < 28:
                    transition['reward'] = 0
                else:
                    transition['reward'] = np.random.randint(-1, 2)

                transition['to'] = int(np.random.choice(possible_states))
                action['transitions'].append(transition)
                possible_states = np.setdiff1d(possible_states,
                                               np.array(transition['to']))

            state['actions'].append(action)

        mdp['states'].append(state)

    return mdp

    #j = json.dumps(mdp, indent=4)
    #print(j)


def binary_level(state):
    if state == 0:
        return 0
    elif 0 < state <= 2:
        return 1
    elif 2 < state <= 6:
        return 2
    elif 7 < state <= 14:
        return 3
    elif 15 < state <= 30:
        return 4
    else:
        return 5


def get_next_possible_states(level):
    if level == 0:
        return np.arange(1, 3)
    elif level == 1:
        return np.arange(3, 7)
    elif level == 2:
        return np.arange(7, 15)
    elif level == 3:
        return np.arange(15, 30)
    else:
        return np.arange(15, 30)


if __name__ == '__main__':
    mdp = generate_mdp(num_states=30, num_transitions=2)
    j = json.dumps(mdp, indent=4)
    #print(j)

    niterations = tester.get_iterations_with_mdptoolbox(mdp)
    print(niterations)



