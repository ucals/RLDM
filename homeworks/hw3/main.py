import numpy as np
import json
import tester


def generate_mdp(num_states=3, num_actions=2, gamma=0.75):
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

            transition = dict()
            transition['id'] = 0
            transition['probability'] = 0.9
            transition['reward'] = np.random.randint(-1, 2)
            transition['to'] = 0
            action['transitions'].append(transition)

            transition = dict()
            transition['id'] = 1
            transition['probability'] = 0.1
            transition['reward'] = 0 if s < 29 else 10
            transition['to'] = s + 1 if s < 29 else 29
            action['transitions'].append(transition)

            state['actions'].append(action)

        mdp['states'].append(state)

    return mdp


if __name__ == '__main__':
    mdp = generate_mdp(num_states=30)
    j = json.dumps(mdp, indent=4)
    #print(j)

    niterations = tester.get_iterations_with_mdptoolbox(mdp)
    print(niterations)



