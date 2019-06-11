import json
import tester
from collections import defaultdict
from tqdm import tqdm


def generate_mdp(gamma=0.75):
    mdp = dict()
    mdp['gamma'] = gamma
    mdp['states'] = list()

    # First state
    s0 = dict()
    s0['id'] = 0
    s0['actions'] = list()
    for a in range(2):
        action = dict()
        action['id'] = a
        action['transitions'] = list()

        transition = dict()
        transition['id'] = 0
        transition['probability'] = 1.0
        transition['reward'] = 0.0
        transition['to'] = a + 1
        action['transitions'].append(transition)

        s0['actions'].append(action)

    mdp['states'].append(s0)

    # From states 1 to 26
    for i in range(1, 27):
        s = dict()
        s['id'] = i
        s['actions'] = list()

        if i % 2 == 1:
            # Deterministic actions
            for a in range(2):
                action = dict()
                action['id'] = a
                action['transitions'] = list()

                transition = dict()
                transition['id'] = 0
                transition['probability'] = 1.0
                transition['reward'] = 0.0
                transition['to'] = i + a + 2
                action['transitions'].append(transition)

                s['actions'].append(action)
        else:
            # Stochastic actions
            for a in range(2):
                action = dict()
                action['id'] = a
                action['transitions'] = list()
                for t in range(2):
                    transition = dict()
                    transition['id'] = t
                    transition['probability'] = 0.5
                    transition['reward'] = 0.0
                    transition['to'] = i + t + 1
                    action['transitions'].append(transition)

                s['actions'].append(action)

        mdp['states'].append(s)

    # State 27, 28 and 29
    s27 = dict()
    s27['id'] = 27
    s27['actions'] = list()
    for a in range(2):
        action = dict()
        action['id'] = a
        action['transitions'] = list()
        transition = dict()
        transition['id'] = 0
        transition['probability'] = 1.0
        transition['reward'] = 1 if a == 0 else -1 #-100.0
        transition['to'] = 0 if a == 0 else 29 #29
        action['transitions'].append(transition)
        s27['actions'].append(action)

    mdp['states'].append(s27)

    s28 = dict()
    s28['id'] = 28
    s28['actions'] = list()
    for a in range(2):
        action = dict()
        action['id'] = a
        action['transitions'] = list()
        transition = dict()
        transition['id'] = 0
        transition['probability'] = 1.0
        transition['reward'] = -1 if a == 0 else 1 #0.0
        transition['to'] = 0 if a == 0 else 29 #29
        action['transitions'].append(transition)
        s28['actions'].append(action)

    mdp['states'].append(s28)

    s29 = dict()
    s29['id'] = 29
    s29['actions'] = list()
    for a in range(2):
        action = dict()
        action['id'] = a
        action['transitions'] = list()
        transition = dict()
        transition['id'] = 0
        transition['probability'] = 1.0
        transition['reward'] = 1.0
        transition['to'] = 29
        action['transitions'].append(transition)
        s29['actions'].append(action)

    mdp['states'].append(s29)

    return mdp


def open_mdp(filename='submitted.json'):
    with open(filename) as f:
        return json.load(f)


def print_mdp(mdp, indent=4):
    j = json.dumps(mdp, indent=indent)
    print(j)


def save_mdp(mdp, filename='solution.json'):
    with open(filename, 'w') as fp:
        json.dump(mdp, fp)


if __name__ == '__main__':
    num_tests = 10
    mdp = generate_mdp()
    #print_mdp(mdp)

    results = defaultdict(lambda: 0)
    for i in tqdm(range(num_tests)):
        niterations = tester.get_iterations_with_mdptoolbox(mdp, verbose=False)
        results[niterations] += 1

    # Printing results
    print('\nResults:')
    for k, v in results.items():
        print(f'- {k} iterations to solve: {v}/{num_tests}')

    save_mdp(mdp)
