# /usr/bin/env python 3.6
# -*-coding:utf-8-*-

'''
tensorflow in cart

Reference link:
https://github.com/gxnk/reinforcement-learning-code/blob/master/

Author: Jing Wang
'''

import numpy as np
import gym
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable

import model


class Cart_Pytorch(object):

	def __init__(self, numFeatures, numActions, learningRate, gamma):
		self.numFeatures = numFeatures
		self.numActions = numActions
		self.lr = learningRate
		self.gamma = gamma

		self.epObs, self.epAs, self.epRs = [],[],[]
		self.actionProb = []
		self.net = self.buildNet()
		self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr)


	def buildNet(self):
		
		return model.PolicyNet(self.numActions)

	def learn(self):
		rewards = self._discountAndReward()
		rewards = torch.Tensor(rewards)
		self.optimizer.zero_grad()
		loss = self._computeLoss(rewards)
		loss.backward()
		self.optimizer.step()
		self.epObs, self.epAs, self.epRs = [],[],[]
		self.actionProb = []
		return rewards.numpy()

	def _computeLoss(self, rewards):
		
		policy_losses = []	

		for log_prob, reward in zip(self.actionProb, rewards):
			policy_losses.append(-log_prob * reward)
		return torch.cat(policy_losses).mean() # torch cat need torch.Tensor

	def _selectAction(self, state):
		state = torch.from_numpy(state[np.newaxis, :]).to(torch.float32)
		state = Variable(state)
		prob = self.net(state)
		prob = Categorical(prob) # for categorical actions

		action  = prob.sample()
		self.actionProb.append(prob.log_prob(action))

		return action.item()

	def greedy(self, state):
		state = torch.from_numpy(state[np.newaxis, :]).to(torch.float32)
		state = Variable(state)
		prob = self.net(state)
		prob = prob.data.numpy()
		action = np.argmax(prob.ravel())
		return action


	def _discountAndReward(self):

		# sum of discounted reward
		discountedRewardSum = np.zeros_like(self.epRs)

		mediateSum = 0
		for t in reversed(range(len(self.epRs))):
			mediateSum = mediateSum * self.gamma + self.epRs[t]
			discountedRewardSum[t] = mediateSum

		# normalize
		discountedRewardSum -= np.mean(discountedRewardSum)
		discountedRewardSum /= np.std(discountedRewardSum)
		return discountedRewardSum

	def storeTrasition(self, state, action, reward):

		self.epObs.append(state)
		self.epAs.append(action)
		self.epRs.append(reward)


if __name__ == '__main__':
	env = gym.make("CartPole-v0").unwrapped
	# env.wrapper()
	
	maxEps = 300
	numFeatures = 4
	numActions = 2
	lr = 0.01
	gamma = 0.95
	pg = Cart_Pytorch(numFeatures, numActions, lr, gamma)

	rewardList = []
	for t in range(maxEps):
		state = env.reset()
		done = False

		rewardSum = 0
		while not done:
			action = pg._selectAction(state)

			newState, reward, done, _ = env.step(action)
			pg.storeTrasition(state, action, reward)
			state = newState

			rewardSum += reward

		discountedReward = pg.learn()
		rewardList.append(rewardSum)

		if t % 100 == 0:
			print("Finish epsilon {}".format(t))

	plt.plot(list(range(maxEps)), rewardList, 'r-', linewidth = 2)
	plt.xlabel("episode")
	plt.ylabel("rewards")
	plt.show()

	state = env.reset()
	done = False
	while not done:
		action = pg.greedy(state)
		state, reward, done, _ = env.step(action)
		env.render()

	env.close()








