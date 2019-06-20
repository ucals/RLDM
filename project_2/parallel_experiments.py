import numpy as np
import os
import json
import multiprocessing as mp
from time import time
from datetime import timedelta
import agent_pytorch as ag


def run_experiment(protocol, experiment, run_filename):
    name = mp.current_process().name

    def epsilon_decay(curr_epsilon, i_episode, min_epsilon=0.05):
        return max(min_epsilon, np.exp(-i_episode / 70))

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

    df_scores.to_csv(run_filename, index=False)
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
    for i, experiment in enumerate(protocol['experiments']):
        t_exp_start = time()
        print('-' * 20)
        print(f'Experiment {experiment["id"]}: {i + 1}/{len(protocol["experiments"])}:')
        print('-' * 20)

        jobs = []
        eid = experiment['id']
        for j in range(protocol['global_params']['runs_per_experiment']):
            run_filename = f'{full_experiment_folder}/df_{eid}_run_{j + 1:02d}.csv'
            job = mp.Process(target=run_experiment, args=(protocol, experiment,
                                                          run_filename),
                             name=f'run_{j + 1:02d}')
            jobs.append(job)

        # Run processes
        for p in jobs:
            p.start()

        # Exit the completed processes
        for p in jobs:
            p.join()

        # TODO combine all DataFrame runs in a single one for the experiment

        print(f'Time to complete experiment {experiment["id"]}:'
              f'{timedelta(seconds=time() - t_exp_start)}\n')

    # TODO combine all DataFrame experiments in a single one for the protocol

    print(f'\nTime to complete all experiments: '
          f'{timedelta(seconds=time() - t_start)}\n')