# /usr/bin/env python 3.6
# -*-coding:utf-8-*-

'''
utility function
'''

import numpy as np 
import scipy.signal as signal
import random
from typing import NamedTuple

def explained_variance_1d(ypred, y):
	"""
	Var[ypred - y] / var[y].
	https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
	"""
	assert y.ndim == 1 and ypred.ndim == 1
	vary = np.var(y)
	return np.nan if vary == 0 else 1 - np.var(y-ypred)/vary


def discount(self, x):
	"""
	Compute discounted sum of future values
	out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
	"""
	return signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]

def flatten(l):
	result = []
	for sub in l:
		for ele in sub:
			result.append(ele)
	return result

def get_discount_reward(rewards, gamma):
	'''
	Get discount reward
	Args:
	rewards (list)
	gamma (float)
	'''
	result = [0] * len(rewards)
	rew = 0
	for i in reversed(range(len(rewards))):
		rew = rew * gamma + rewards[i]
		result[i] = rew
	return result

# make transition
class Transition(NamedTuple):
	state: np.array
	action: np.array or int
	next_state: np.array
	reward: float

# define memory replay
class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity 
		self.memory = []

	def push(self, transition):
		if len(self.memory) == self.capacity:
			self.memory.pop(0)
		self.memory.append(transition)

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

	def clear(self):
		self.memory = []
