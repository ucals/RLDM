import pandas as pd
import numpy as np
import os
import json
import multiprocessing as mp
import random
import string
from time import time
from time import sleep
from datetime import timedelta
import agent_pytorch as ag


def run_experiment(protocol, experiment, run_filename, frames):
    name = mp.current_process().name

    def epsilon_decay(curr_epsilon, i_episode, min_epsilon=0.05):
        return max(min_epsilon, np.exp(-i_episode / 70))

    t_run_start = time()
    print(f'Starting {name}/{protocol["global_params"]["runs_per_experiment"]}, '
          f'experiment {experiment["id"]} ({i + 1}/{len(protocol["experiments"])})')

    agent = ag.Agent(batch_size=protocol['global_params']['batch_size'],
                     layers=protocol['global_params']['layers'],
                     alpha=protocol['global_params']['alpha'],
                     dueling=experiment['dueling'],
                     double=experiment['double'],
                     prioritized_er=experiment['prioritized_er'])

    df_scores = agent.train(run_n=name,
                            max_episodes=protocol['global_params']['max_episodes'],
                            stop_when_solved=protocol['global_params']['stop_when_solved'],
                            keep_learning=protocol['global_params']['keep_learning'],
                            epsilon_decay=epsilon_decay,
                            print_frequency=protocol['global_params']['print_frequency'],
                            render=False,
                            print_same_line=False,
                            log_floydhub=False)

    df_scores.to_csv(run_filename, index=False)
    df_scores.columns = [f'{c}_{experiment["id"]}_{name}'
                         for c in df_scores.columns]
    frames.put(df_scores)
    print(f'{name}\tTime to complete: {timedelta(seconds=time() - t_run_start)}\n')


if __name__ == '__main__':
    experiment_name = 'e1'

    base_experiments_folder = 'experiments'
    full_experiment_folder = f'{base_experiments_folder}/{experiment_name}'

    if not os.path.exists(full_experiment_folder):
        os.makedirs(full_experiment_folder)

    protocol_filename = f'experiments/{experiment_name}/protocol.json'
    with open(protocol_filename, 'r') as f:
        protocol = json.load(f)

    t_start = time()
    frames = mp.Queue()
    for i, experiment in enumerate(protocol['experiments']):
        jobs = []
        eid = experiment['id']
        for j in range(protocol['global_params']['runs_per_experiment']):
            run_filename = f'{full_experiment_folder}/df_{eid}_run_{j + 1:02d}.csv'
            job = mp.Process(target=run_experiment, args=(protocol, experiment,
                                                          run_filename,
                                                          frames),
                             name=f'run_{j + 1:02d}')
            jobs.append(job)

        # Run processes
        for p in jobs:
            p.start()

        # Exit the completed processes
        for p in jobs:
            p.join()

        experiment_filename = f'{full_experiment_folder}/df_{eid}.csv'
        df_experiment = pd.concat([frames.get() for p in jobs], axis=1)
        df_experiment.to_csv(experiment_filename, index=False)

        break

    print(f'\nTime to complete all experiments: '
          f'{timedelta(seconds=time() - t_start)}\n')
