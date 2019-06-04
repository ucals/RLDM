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
            possible_states = np.arange(num_states)
            for t in range(num_transitions):
                transition = dict()
                transition['id'] = t
                transition['probability'] = 0.5
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


if __name__ == '__main__':
    #with open('tester/sample.json') as data_file:
    #    mdp = json.load(data_file)
    #niterations = tester.get_iterations_with_mdptoolbox(mdp)
    #print(niterations)

    mdp = generate_mdp(num_states=30)
    j = json.dumps(mdp, indent=4)
    #print(j)

    niterations = tester.get_iterations_with_mdptoolbox(mdp)
    print(niterations)



