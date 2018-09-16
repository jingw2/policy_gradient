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


class PolicyGradient(object):

	def __init__(self, state_size, action_size, lr, gamma):
		self.state_size = state_size
		self.action_size = action_size
		self.lr = lr
		self.gamma = gamma

		self.observations, self.actions, self.rewards = [],[],[]
		self.action_probs = []
		self.net = model.Actor(self.state_size, self.action_size)
		self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr)


	def learn(self):
		rewards = self.get_discounted_reward()
		rewards = torch.Tensor(rewards)
		self.optimizer.zero_grad()
		loss = self.get_loss(rewards)
		loss.backward()
		self.optimizer.step()
		self.observations, self.actions, self.rewards = [],[],[]
		self.action_probs = []
		return rewards.numpy()

	def get_loss(self, rewards):
		
		policy_losses = []	

		for log_prob, reward in zip(self.action_probs, rewards):
			policy_losses.append(-log_prob * reward)
		return torch.cat(policy_losses).mean() # torch cat need torch.Tensor

	def select_action(self, state):
		state = torch.from_numpy(state[np.newaxis, :]).to(torch.float32)
		state = Variable(state)
		prob = self.net(state)
		prob = Categorical(prob) # for categorical actions

		action  = prob.sample()
		self.action_probs.append(prob.log_prob(action))

		return action.item()

	def greedy(self, state):
		state = torch.from_numpy(state[np.newaxis, :]).to(torch.float32)
		state = Variable(state)
		prob = self.net(state)
		prob = prob.data.numpy()
		action = np.argmax(prob.ravel())
		return action


	def get_discounted_reward(self):

		# sum of discounted reward
		discounted_reward = np.zeros_like(self.rewards)

		mediate_sum = 0
		for t in reversed(range(len(self.rewards))):
			mediate_sum = mediate_sum * self.gamma + self.rewards[t]
			discounted_reward[t] = mediate_sum

		# normalize
		discounted_reward -= np.mean(discounted_reward)
		discounted_reward /= np.std(discounted_reward)
		return discounted_reward

	def store_transition(self, state, action, reward):

		self.observations.append(state)
		self.actions.append(action)
		self.rewards.append(reward)


if __name__ == '__main__':
	env = gym.make("CartPole-v0")
	# env.wrapper()
	
	max_episodes = 800
	global state_size, action_size
	state_size = int(np.product(env.observation_space.shape))
	action_size = int(env.action_space.n)
	lr = 0.001
	gamma = 0.99
	pg = PolicyGradient(state_size, action_size, lr, gamma)

	reward_list = []
	for t in range(max_episodes):
		state = env.reset()
		done = False

		reward_sum = 0
		while not done:
			action = pg.select_action(state)

			next_state, reward, done, _ = env.step(action)
			pg.store_transition(state, action, reward)
			state = next_state

			reward_sum += reward

		discounted_reward = pg.learn()
		reward_list.append(reward_sum)

		average_reward = np.mean(reward_list[-100:])
		if t % 100 == 0:
			print("Iteration: {}, last 100 average reward: {}, average reward: {}".format(t + 1, \
				average_reward, np.mean(reward_list)))

		if average_reward > env.spec.reward_threshold:
			print("Solved!")
			break

	running_rewards = reward_list
	rewards = np.array(running_rewards)
	rewards_mean = np.mean(rewards)
	rewards_std = np.std(rewards)

	plt.plot(running_rewards)
	plt.fill_between(
	    range(len(rewards)),
	    rewards-rewards_std, 
	    rewards+rewards_std, 
	    color='orange', 
	    alpha=0.2
	)
	plt.title(
	    'Vanilla Policy Gradient Rewards Mean: {:.2f}, Standard Deviation: {:.2f}'.format(
	        np.mean(running_rewards),
	        np.std(running_rewards)
	    )
	)
	plt.show()








