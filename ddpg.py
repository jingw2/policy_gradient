# /usr/bin/env python 3.6
# -*-coding:utf-8-*-

'''
Deep Deterministic Policy Gradient (DDPG)

Reference Link:

Author: Jing Wang
'''

# set up
import numpy as np
import util
import model
import gym
import matplotlib.pyplot as plt 
import random
from copy import deepcopy

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

CAPACITY = 100000
BATCHSIZE = 128
GAMMA = 0.99
# set device cpu or gpu
random.seed(100)

replay_memory = util.ReplayMemory(CAPACITY)

class DDPG(object):

	def __init__(self, env, actor, critic, target_actor, target_critic, num_episode, replay_memory, gamma, lr = 0.001):
		self.env = env
		self.actor = actor
		self.critic = critic
		self.target_actor = target_actor
		self.target_critic = target_critic
		self.num_episode = num_episode
		self.gamma = gamma	
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = 1e-3)
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = 1e-4)
		self.replay_memory = replay_memory
		self.lr = lr
		self.tau = 0.001
		self.loss_func = nn.MSELoss()

	def step(self):
		'''Optimize step'''
		
		reward_list = []
		for i in range(self.num_episode):
			state = self.env.reset()
			# random exploration noise
			done = False
			reward_sum = 0
			# noise = np.random.random(action_size)
			noise = np.zeros(action_size)
			while not done:
				# select action
				action = self.select_action(state)
				selected_action = np.argmax(action + noise)
				
				new_state, reward, done, _ = self.env.step(selected_action)

				# push memory
				if done:
					new_state = None
				self.replay_memory.push(util.Transition(state, action, new_state, reward))
				state = new_state
				reward_sum += reward
			reward_list.append(reward_sum)

			for _ in range(20):
				self.train()

			average_reward = np.mean(reward_list[-100:])
			if i % 100 == 0:
				print("Iteration: {}, last 100 average reward: {}, average reward: {}".format(i + 1, \
					average_reward, np.mean(reward_list)))

			if average_reward > self.env.spec.reward_threshold:
				print("Solved!")
				break

		return reward_list

	def select_action(self, state):
		state = torch.Tensor(state).float().unsqueeze(0)
		action_scores = self.actor(state)
		return action_scores.data.numpy()


	def train(self):
		if len(self.replay_memory) < BATCHSIZE:
			return

		# transitions = memory.sample(BATCHSIZE)
		batch = self.replay_memory.sample(BATCHSIZE)
		# batch = util.Transition(*zip(*transitions))

		not_end_index = torch.tensor(tuple(map(lambda s: 
			s.next_state is not None, batch)), 
			dtype = torch.uint8)

		not_end = torch.cat([torch.Tensor(b.next_state).float() for b in batch
			if b.next_state is not None]).view(-1, state_size)

		state_batch= torch.cat([torch.Tensor(b.state).float() for b in batch]).view(-1, state_size)
		action_batch = torch.cat([torch.Tensor(b.action).float() for b in batch]).view(-1, action_size)
		reward_batch = torch.cat([torch.Tensor([b.reward]).float() for b in batch])

		# print("action batch: ", action_batch.unsqueeze(1))
		# print("state batch size: ", state_batch.size())
		# get q value by correspondent action position
		# print(action_batch.long().view(-1).size())
		# print(self.critic([state_batch, action_batch]).size())
		Qval = self.critic([state_batch, action_batch]).view(-1)

		# get new q value
		new_Qval = torch.zeros_like(not_end_index, dtype = torch.float) # by default, 
								# the q-value of ending state is 0

		next_action_batch = self.target_actor(not_end).view(-1, action_size).float()
		new_Qval[not_end_index] = self.target_critic([not_end, next_action_batch]).view(-1).detach() # detach copy

		target = new_Qval * self.gamma + reward_batch


		# compute loss using Huber loss
		# reference: https://en.wikipedia.org/wiki/Huber_loss
		# newQval shape (BATCHSIZE,), should add one dimension, unsqueeze 1
		

		# clear gradients
		self.critic_optimizer.zero_grad()
		loss = self.loss_func(Qval, target)
		# self.critic.zero_grad()
		loss.backward()
		self.critic_optimizer.step()

		# update actor
		# self.actor.zero_grad()
		self.actor_optimizer.zero_grad()
		actions = self.actor(state_batch).view(-1, action_size).float()
		policy_loss = - self.critic([state_batch, actions]).mean()
		
		policy_loss.backward()
		self.actor_optimizer.step()


		self.target_actor = self.update_target(self.actor, self.target_actor)
		self.target_critic = self.update_target(self.critic, self.target_critic)

	
	def update_target(self, source, target):
		new_target_param = parameters_to_vector(source.parameters()) * self.tau + \
				(1 - self.tau) * parameters_to_vector(target.parameters())
		vector_to_parameters(new_target_param, target.parameters())
		return target

if __name__ == '__main__':
	env = gym.make("CartPole-v0")

	global state_size, action_size
	state_size = int(np.product(env.observation_space.shape))
	action_size = int(env.action_space.n)
	num_episode = 800
	critic = model.Critic(state_size, action_size)
	actor = model.Actor(state_size, action_size)

	# actor.eval()
	# critic.eval()

	# target network
	target_critic = deepcopy(critic)
	target_actor = deepcopy(actor)

	ddpg = DDPG(env, actor, critic, target_actor, target_critic, num_episode, replay_memory, gamma = 0.99)
	running_rewards = ddpg.step()
	
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
	    'DDPG Rewards Mean: {:.2f}, Standard Deviation: {:.2f}'.format(
	        np.mean(running_rewards),
	        np.std(running_rewards)
	    )
	)
	plt.show()

	