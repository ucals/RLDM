# -*- coding: utf-8 -*-
"""Soccer Environment module.

This module implements Soccer environment as described in Greenwald & Hall's
`Correlated-Q Learning paper`_. Implementation follows `OpenAI Gym standards`_.

.. _Correlated-Q Learning paper:
   https://www.aaai.org/Papers/ICML/2003/ICML03-034.pdf

.. _OpenAI Gym standards:
   http://gym.openai.com/docs/

"""

import numpy as np
import random


class Soccer(object):
    """Soccer Environment.

    This class implements Soccer environment. All attributes are private.
    Game simulation should be done using the methods, which return the current
    state. The current state is a tuple containing:
    - Position of Player 0 [row, col] in 2x4 grid
    - Position of Player 1 [row, col] in 2x4 grid
    - Who is in possession of the ball (0 or 1)

    Attributes:
        num_actions (int): Number of possible actions in the environment (5).

    """
    def __init__(self, seed=None):
        """Initializes the environment.

        Args:
            seed (int): Random seed. If set, will set Numpy and Random seeds, to
                enable reproducibility of results.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.num_actions = 5
        self._pos_p0 = [None, None]
        self._pos_p1 = [None, None]
        self._ball = None
        self._t = None
        self.reset()

    def step(self, action_p0, action_p1):
        """Simulates a step given joint actions from players 0 and 1, returning
        the new state.

        Args:
            action_p0 (int): Action from player 0: 1 for North, 2 for East, 3
                for South, 4 for West, 0 for stay put.
            action_p1 (int): Action from player 1: 1 for North, 2 for East, 3
                for South, 4 for West, 0 for stay put.

        Returns:
            List containing current state components.
        """
        self._t += 1

        first_to_act = np.random.randint(2)  # Defines who acts first

        # Updates players positions, and defines ball possession
        if first_to_act == 0:
            possible_pos_p0 = self._get_new_position(self._pos_p0, action_p0)
            if possible_pos_p0 != self._pos_p1:
                self._pos_p0 = possible_pos_p0

            possible_pos_p1 = self._get_new_position(self._pos_p1, action_p1)
            if possible_pos_p1 != self._pos_p0:
                self._pos_p1 = possible_pos_p1
            else:
                if self._ball == 1:
                    self._ball = 0
        else:
            possible_pos_p1 = self._get_new_position(self._pos_p1, action_p1)
            if possible_pos_p1 != self._pos_p0:
                self._pos_p1 = possible_pos_p1

            possible_pos_p0 = self._get_new_position(self._pos_p0, action_p0)
            if possible_pos_p0 != self._pos_p1:
                self._pos_p0 = possible_pos_p0
            else:
                if self._ball == 0:
                    self._ball = 1

        # Check whether a goal was scored to define rewards
        reward_p0 = 0
        reward_p1 = 0
        done = False
        if self._ball == 0:
            if self._pos_p0[1] == 0:
                reward_p0 = -100
                reward_p1 = 100
                done = True
            elif self._pos_p0[1] == 3:
                reward_p0 = 100
                reward_p1 = -100
                done = True

        elif self._ball == 1:
            if self._pos_p1[1] == 0:
                reward_p0 = -100
                reward_p1 = 100
                done = True
            elif self._pos_p1[1] == 3:
                reward_p0 = 100
                reward_p1 = -100
                done = True

        return [self._pos_p0, self._pos_p1, self._ball], \
               [reward_p0, reward_p1], done, [first_to_act, self._t]

    def _get_new_position(self, curr_pos, action):
        """Updates current position given an action.

        Args:
            curr_pos (list): List [row, col] in 2x4 grid.
            action (int): Action 1 for North, 2 for East, 3 for South, 4 for
            West, 0 for stay put.

        Returns:
            List of new coordinates
        """
        if action == 0:
            return curr_pos  # Stay put
        elif curr_pos[0] == 0 and action == 1:
            return curr_pos  # Boundary condition
        elif curr_pos[0] == 1 and action == 3:
            return curr_pos  # Boundary condition
        elif curr_pos[1] == 0 and action == 4:
            return curr_pos  # Boundary condition
        elif curr_pos[1] == 3 and action == 2:
            return curr_pos  # Boundary condition

        if action == 1:
            return [0, curr_pos[1]]
        elif action == 2:
            return [curr_pos[0], curr_pos[1] + 1]
        elif action == 3:
            return [1, curr_pos[1]]
        elif action == 4:
            return [curr_pos[0], curr_pos[1] - 1]

    def reset(self):
        """Resets the environment.

        This method must be called after a goal is scored, which is signaled by
        done = True return in step method.

        Returns:
            List containing current state components.

        """
        self._pos_p0 = [0, 1]
        self._pos_p1 = [0, 2]
        self._ball = np.random.randint(2)
        self._t = 0
        return [self._pos_p0, self._pos_p1, self._ball]

    def render(self):
        """Renders the environment in screen, for visualization/debugging
        purpose.

        """
        s = np.full((2, 4), '   ')
        if self._ball == 0:
            s_p0 = ' 0*'
            s_p1 = ' 1 '
        elif self._ball == 1:
            s_p0 = ' 0 '
            s_p1 = '*1 '

        s[self._pos_p0[0]][self._pos_p0[1]] = s_p0
        s[self._pos_p1[0]][self._pos_p1[1]] = s_p1

        print(u'\u250c\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2510')
        print(u'\u2502' + s[0][0] + u'\u2502' + s[0][1] + u'\u2502' + s[0][2] + u'\u2502' + s[0][3] + u'\u2502')
        print(u'\u251c\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2524')
        print(u'\u2502' + s[1][0] + u'\u2502' + s[1][1] + u'\u2502' + s[1][2] + u'\u2502' + s[1][3] + u'\u2502')
        print(u'\u2514\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2518')


if __name__ == '__main__':
    env = Soccer()
    observation = env.reset()
    done = False
    while not done:
        env.render()
        print(observation)
        a0 = np.random.randint(env.num_actions)
        a1 = np.random.randint(env.num_actions)
        observation, rewards, done, info = env.step(a0, a1)
        print(f'Actions: {[a0, a1]}, Rewards: {rewards}, Done: {done}, Info: {info}')
        if done:
            env.render()
            print('DONE!')
