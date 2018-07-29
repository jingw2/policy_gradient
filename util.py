# /usr/bin/env python 3.6
# -*-coding:utf-8-*-

'''
utility function
'''

import numpy as np 
import scipy.signal as signal

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
	result = []
	rew = 0
	for power, r in enumerate(rewards):
		rew += r * gamma ** power
		result.append(rew)
	result.reverse()
	return result