# /usr/bin/env python 3.6
# -*-coding:utf-8-*-

'''
Proximal Policy Optimization

Author: Jing Wang
'''

import math
import argparse
import collections
from itertools import count
import numpy as np
import gym
from copy import deepcopy
import matplotlib.pyplot as plt 

import torch
from torch.autograd import Variable
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch.distributions import Multinomial, Categorical

import util
import model

parser = argparse.ArgumentParser(description='different loss function')
parser.add_argument('--loss_func', type=str, default = "clip_surrogate",
                    help='loss function choice')
parser.add_argument('--epsilon', type=float, default = 0.2,
                    help='clipped hyperparameter')
parser.add_argument('--beta', type=float, default = 3,
                    help='kl hyperparameter')
parser.add_argument('--d_target', type=float, default = 0.001,
                    help='kl hyperparameter')
args = parser.parse_args()

class PPO(object):

	def __init__(self, actor, value_net, env, lr = 1e-3, epsilon = 0.2, beta = 3, d_target = 0.001, loss_choice = "clip_surrogate"):
		'''
		Args:
		actor (actor network)
		value_net (value network)
		lr (float): learning rate
		epsilon (float): clipping hyperparameter
		beta (float): adaptive kl divergence objective hyperparameter
		loss_choice (str): different loss function choice
		'''
		self.actor = actor
		self.old_model = deepcopy(actor)
		self.env = env
		self.value_net = value_net
		self.epsilon = epsilon
		self.loss_choice = loss_choice
		self.optimizer = optim.Adam(list(self.actor.parameters()) + \
			list(self.value_net.parameters()), lr = lr)
		self.loss_func = nn.MSELoss()
		self.gamma = 0.99
		self.d_target = d_target
		self.beta = beta
			

	def collect_data(self):
		roll_out = util.ReplayMemory(1e6)

		state = self.env.reset()
		done = False
		while not done:
			state_tensor = torch.Tensor(state).float().unsqueeze(0)
			action_probs = self.old_model(state_tensor)
			probs = Categorical(action_probs)
			action = probs.sample().item()
			next_state, reward, done, info = self.env.step(action)
			if done:
				next_state = None
			roll_out.push(util.Transition(state, action, next_state, reward))
			state = next_state
		return roll_out.memory


	def train(self, roll_out):

		states = torch.cat([torch.Tensor(b.state).float() for b in roll_out]).view(-1, state_size)
		actions = torch.cat([torch.Tensor([b.action]).long() for b in roll_out]).view(-1, 1)
		rewards = [b.reward for b in roll_out]

		discounted_rewards = util.get_discount_reward(rewards, self.gamma)
		# to tensor
		discounted_rewards = torch.Tensor(discounted_rewards).float().view(-1)
		qvalue = self.value_net(states).view(-1)

		advantage = discounted_rewards - qvalue

		value_loss = self.loss_func(qvalue, discounted_rewards)

		# calculate distribution ratio
		with torch.no_grad():
			prob_old = self.old_model(states).gather(1, actions)
		prob = self.actor(states).gather(1, actions)

		ratio = (prob / prob_old).view(-1)

		# clipped surrogate objective
		if self.loss_choice == "clip_surrogate":
			policy_loss = torch.min(torch.stack([- (ratio * advantage), - (ratio.clamp(1 - self.epsilon,\
						 1 + self.epsilon) * advantage)], dim = 1), dim = 1)[0].mean()
		elif self.loss_choice == "fixed_kl":
			kl_divergence = torch.sum(prob_old * torch.log(prob / prob_old), dim = 1).view(-1)
			policy_loss = -(ratio * advantage - self.beta * kl_divergence).mean()
		elif self.loss_choice == "adaptive_kl":
			kl_divergence = torch.sum(prob_old * torch.log(prob / prob_old), dim = 1).view(-1)
			d = kl_divergence.mean()
			if d < self.d_target / 1.5:
				self.beta /= 2
			if d > self.d_target * 1.5:
				self.beta *= 2
			policy_loss = -(ratio * advantage - self.beta * kl_divergence).mean()


		loss = policy_loss + value_loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


	def step(self, num_episodes):
		running_rewards = []
		for t in range(num_episodes):
			roll_out = [self.collect_data() for _ in range(4)]

			for data in roll_out:
				self.train(data)

			reward_sum = sum([d.reward for d in data])
			running_rewards.append(reward_sum)

			self.old_model = deepcopy(self.actor)

			average_rewards = np.mean(running_rewards[-100:])

			if t % 100 == 0:
				print("Episode: {}, last 100 episodes rewards: {}, average rewards: {}".format(t, \
					average_rewards, np.mean(running_rewards)))

			if average_rewards > self.env.spec.reward_threshold:
				print("Solved!")
				break
		return running_rewards



if __name__ == '__main__':
	env = gym.make("CartPole-v0")

	global state_size, action_size
	state_size = int(np.product(env.observation_space.shape))
	action_size = int(env.action_space.n)
	num_episodes = 800

	actor = model.Actor(state_size, action_size)
	value_net = model.ValueNet(state_size)

	loss_choice = args.loss_func
	beta = args.beta
	epsilon = args.epsilon
	d_target = args.d_target
	ppo = PPO(actor, value_net, env, epsilon = epsilon, beta = beta, d_target = d_target, loss_choice = loss_choice)
	running_rewards = ppo.step(num_episodes)

	# plot
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
	    'PPO {} : Rewards Mean: {:.2f}, Standard Deviation: {:.2f}'.format(loss_choice, 
	        np.mean(running_rewards),
	        np.std(running_rewards)
	    )
	)
	plt.show()