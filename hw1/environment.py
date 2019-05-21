import numpy as np
from random import randint
from numpy.random import choice
import pprint


class Environment(object):
	def __init__(self, mask):
		self.mask = mask
		self.bankroll = 0
		self.count = 0
		self.num_actions = 2

	def __str__(self):
		return f'N = {len(self.mask)}, isBadSide = {self.mask}'
		
	def step(self, action):
		"""Step function

		Args:
			action: The action being taken

		Returns:
			tuple: Containing next state, reward, done and info

		"""
		self.count += 1
		if action == 0:
			return self.bankroll, 0, True, (self.count, None)
		
		roll = randint(1, len(self.mask))
		if self.mask[roll - 1] == 1:
			return 0, -self.bankroll, True, (self.count, roll)
		else:
			self.bankroll += roll
			return self.bankroll, roll, False, (self.count, roll)
		
	def reset(self):
		self.bankroll = 0
		self.count = 0
		return self.bankroll

	def get_probs(self, state, action):
		"""Get transition probabilities after taking action in a given state

		Args:
			state: The current state
			action: The action taken

		Returns:
			list of tuples: A list of tuples containing probability, next_state,
				reward and done

		"""
		if self.mask[state - 1] == 1:
			return [(1, 0, -state, True)]

		if action == 0:
			return [(1, state, 0, True)]

		result = []
		for roll in range(1, len(self.mask) + 1):
			if self.mask[roll - 1] == 1:
				leaf = (1.0 / len(self.mask), 0, -state, True)
			else:
				leaf = (1.0 / len(self.mask), state + roll, roll, False)

			result.append(leaf)

		return result
		
		
if __name__ == '__main__':
	pp = pprint.PrettyPrinter()
	env = Environment([1, 1, 1, 0, 0, 0])
	pp.pprint(env.get_probs(0, 1))

	exit(0)

	for i_episode in range(20):
		print(f'Episode {i_episode + 1}')
		state = env.reset()
		done = False
		while not done:
			action = choice([0, 1], p=[0.1, 0.9])
			next_state, reward, done, info = env.step(action)
			print(state, action, next_state, reward, done, info)
			state = next_state
			
		print(' ')

