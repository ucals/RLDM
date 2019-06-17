#!/anaconda3/bin/python

# To run in Floydhub:
# floyd run --cpu 'python main.py --fh'

import numpy as np
import agent as ag
import argparse
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

    agent = ag.Agent(batch_size=32, layers=[512, 512])
    print(agent.Q.summary())

    def epsilon_decay1(curr_epsilon, i_episode, min_epsilon=0.05, decay=0.999):
        return max(min_epsilon, curr_epsilon * decay)

    def epsilon_decay2(curr_epsilon, i_episode, min_epsilon=0.05):
        return max(min_epsilon, np.exp(-i_episode / 70))

    print('Agent parameters: ')
    print(vars(agent))
    print(' ')

    agent.train(epsilon_decay=epsilon_decay2, render=render,
                print_same_line=not args.fh, log_floydhub=args.fh)
