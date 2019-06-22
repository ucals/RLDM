#!/anaconda3/bin/python

# To run in Floydhub:
# floyd run --cpu 'python main.py --fh'

import numpy as np
import agent_pytorch as ag
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DoubleDQNetwork with '
                                                 'Experience Replay.')
    parser.add_argument('-f', '--fh', action='store_true',
                        help='log in Floydhub format')
    parser.add_argument('-r', '--render', action='store_true',
                        help='render animation in screen')
    args = parser.parse_args()

    render = args.render
    if args.fh:
        render = False

    agent = ag.Agent(batch_size=32, layers=[512, 512], dueling=True,   #batch_size=32
                     double=True, alpha=0.0001, prioritized_er=True)  # alpha=0.00025

    def epsilon_decay1(curr_epsilon, i_episode, min_epsilon=0.05, decay=0.999):
        return max(min_epsilon, curr_epsilon * decay)

    def epsilon_decay2(curr_epsilon, i_episode, min_epsilon=0.05):
        return max(min_epsilon, np.exp(-i_episode / 70))

    print('Agent parameters: ')
    print(vars(agent))
    print('\n')

    agent.train(epsilon_decay=epsilon_decay2, render=render, print_frequency=10,
                print_same_line=False, log_floydhub=args.fh,  #  print_same_line=not args.fh
                score_filename='live_score1.csv', stop_when_solved=True,
                keep_learning=False, max_episodes=1000, vectorized=False)
