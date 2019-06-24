import pandas as pd
import numpy as np
import os
import json
import multiprocessing as mp
from time import time
from datetime import timedelta
import argparse
import agent_pytorch as ag
from epsilon_curves import get_epsilon_decay


def run_experiment(protocol, experiment, folder):
    name = mp.current_process().name

    decay_function = experiment['decay_function'] if 'decay_function' in \
                                                     experiment else 0
    epsilon_decay = get_epsilon_decay(decay_function=decay_function)

    t_run_start = time()
    print(f'Starting {name}/{protocol["global_params"]["runs_per_experiment"]}, '
          f'experiment {experiment["id"]} ({i + 1}/{len(protocol["experiments"])})')

    agent = ag.Agent(batch_size=protocol['global_params']['batch_size'],
                     layers=protocol['global_params']['layers'],
                     alpha=experiment['alpha'],
                     gamma=experiment['gamma'],
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
    run_filename = f'{folder}/df_{eid}_run_{j + 1:02d}.csv'
    df_scores.to_csv(run_filename, index=False)

    if 'test_for' in protocol['global_params']:
        df_scores_test = agent.test(run_n=name,
                                    max_episodes=protocol['global_params']['test_for'],
                                    print_frequency=protocol['global_params']['print_frequency'],
                                    render=False,
                                    print_same_line=False)
        test_run_filename = f'{folder}/df_{eid}_run_{j + 1:02d}_test.csv'
        df_scores_test.to_csv(test_run_filename, index=False)

    print(f'{name}\tTime to complete: {timedelta(seconds=time() - t_run_start)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Lunar Lander solver '
                                                 'experiments in parallel.')
    parser.add_argument('experiment', help='experiment name in experiments '
                                           'folder', type=str)
    args = parser.parse_args()

    experiment_name = args.experiment

    base_experiments_folder = 'experiments'
    full_experiment_folder = f'{base_experiments_folder}/{experiment_name}'

    if not os.path.exists(full_experiment_folder):
        os.makedirs(full_experiment_folder)

    protocol_filename = f'experiments/{experiment_name}/protocol.json'
    with open(protocol_filename, 'r') as f:
        protocol = json.load(f)

    t_start = time()
    jobs = []
    for i, experiment in enumerate(protocol['experiments']):
        t_exp_start = time()
        print('-' * 20)
        print(f'Protocol: {protocol_filename}')
        print(f'Experiment {experiment["id"]}: {i + 1}/{len(protocol["experiments"])}:')
        print('-' * 20)
        job = mp.Process(target=run_experiment, args=(protocol, experiment,
                                                      full_experiment_folder),
                         name=experiment['id'])
        jobs.append(job)

    # Run processes
    for p in jobs:
        p.start()

    # Exit the completed processes
    for p in jobs:
        p.join()

    print(f'\nTime to complete all experiments: '
          f'{timedelta(seconds=time() - t_start)}\n')