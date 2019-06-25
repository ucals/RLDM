import numpy as np
import json
import os


def generate_json(experiment, num_alphas=31):
    protocol = {
        'experiments': [

        ],
        'global_params': {
            'optimizer': 'rmsprop',
            'batch_size': 32,
            'layers': [512, 512],
            'max_episodes': 1000,
            'stop_when_solved': False,
            'keep_learning': True,
            'print_frequency': 50
        }
    }
    a = np.logspace(start=-6, stop=-1, num=num_alphas, base=10)
    for i in range(num_alphas):
        e = {"id": f'E{i + 1:02d}', "dueling": True, "double": True,
             "prioritized_er": True, "alpha": a[i], "gamma": 0.99}
        protocol['experiments'].append(e)

    if not os.path.exists(f'experiments/{experiment}'):
        os.makedirs(f'experiments/{experiment}')

    with open(f'experiments/{experiment}/protocol.json', 'w') as fp:
        json.dump(protocol, fp, indent=4)


if __name__ == '__main__':
    generate_json(experiment='e5a')
