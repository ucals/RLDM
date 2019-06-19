#!/anaconda3/bin/python

import numpy as np
import pandas as pd
import json
import os
from time import time
from datetime import timedelta
import agent_pytorch as ag


if __name__ == '__main__':
    experiment_name = 'e1'

    def epsilon_decay(curr_epsilon, i_episode, min_epsilon=0.05):
        return max(min_epsilon, np.exp(-i_episode / 70))

    protocol = {
        'experiments': [
            {'id': 'a', 'dueling': True, 'double': True, 'prioritized_er': True},
            {'id': 'b', 'dueling': False, 'double': False, 'prioritized_er': False},
            {'id': 'c', 'dueling': True, 'double': False, 'prioritized_er': False},
            {'id': 'd', 'dueling': False, 'double': True, 'prioritized_er': False},
            {'id': 'e', 'dueling': False, 'double': False, 'prioritized_er': True}
        ],
        'global_params': {
            'runs_per_experiment': 20,
            'batch_size': 32,
            'layers': [512, 512],
            'alpha': 0.0001,
            'max_episodes': 500,
            'stop_when_solved': False
        }
    }

    if not os.path.exists('experiments'):
        os.makedirs('experiments')

    with open(f'experiments/{experiment_name}.json', 'w') as fp:
        json.dump(protocol, fp, indent=4)

    frames = []
    t_start = time()
    for i, experiment in enumerate(protocol['experiments']):
        for j in range(protocol['global_params']['runs_per_experiment']):
            t_run_start = time()
            print(f'Experiment {experiment["id"]}: {i + 1}/{len(protocol["experiments"])}:')
            print(f'Run {j + 1}/{protocol["global_params"]["runs_per_experiment"]}:')

            agent = ag.Agent(batch_size=protocol['global_params']['batch_size'],
                             layers=protocol['global_params']['layers'],
                             alpha=protocol['global_params']['alpha'],
                             dueling=experiment['dueling'],
                             double=experiment['double'],
                             prioritized_er=experiment['prioritized_er'])

            df_scores = agent.train(max_episodes=protocol['global_params']['max_episodes'],
                                    stop_when_solved=protocol['global_params']['stop_when_solved'],
                                    epsilon_decay=epsilon_decay,
                                    print_frequency=50,
                                    render=False,
                                    print_same_line=False,
                                    log_floydhub=False)

            df_scores.columns = [f'{c}_{experiment["id"]}_run_{j + 1}'
                                 for c in df_scores.columns]
            frames.append(df_scores)
            print(f'Time to complete run: {timedelta(seconds=time() - t_run_start)}\n')

    df_experiment = pd.concat(frames, axis=1)
    df_experiment.to_csv(f'experiments/{experiment_name}.csv')
    print(f'Time to complete experiment: {timedelta(seconds=time() - t_start)}')
