# /usr/bin/env python 3.6
# -*-coding:utf-8-*-

'''
Trust Region Policy Optimization

Author: Jing Wang
'''

import math
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
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch.distributions import Multinomial, Categorical

import util
import model


class TRPO(object):

	def __init__(self, policyNet, valueFunc, env, method = "single_path"):
		self.cg_iter = 10
		self.pg_model = policyNet
		self.val_model = valueFunc
		self.max_episode_num = 1
		self.env = env
		self.gamma = 0.95
		self.damping = 0.001
		self.delta = 0.01
		self.advantage = None
		self.batch_size = 32 # for image training
		self.lamb = 0.0

	##########################################################
	# Step 1: 
	# sample actions and trajectories
	##########################################################

	def sample_action_from_policy(self, state):
		'''
		sample action by policy gradient model using multinomial distribution
		'''
		# for single path, sample number is 1
		num_samples = 1
		# to tensor
		state_tensor = torch.Tensor(state).float().unsqueeze(0)

		# get probabilities
		probs = self.pg_model(Variable(state_tensor, requires_grad = True))
		prob_dist = Categorical(probs)
		action = prob_dist.sample()

		return action, probs

	def sample_trajectory(self):
		'''
		sample a sequence of trajectories
		'''
		# initialize
		trajectories = []
		counter = 0
		entropy = 0 # entropy 

		num_actions = self.env.action_space.n

		for _ in range(self.max_episode_num):
			states, actions, rewards, action_dist = [], [], [], []
			state = self.env.reset()
			while True:
				states.append(state)

				# get action
				action, dist = self.sample_action_from_policy(state)

				# step 
				newState, reward, done, _ = self.env.step(action.item())

				# compute entropy 
				entropy += - (dist * dist.log()).sum()

				state = newState
				actions.append(action)
				rewards.append(reward)

				action_dist.append(dist)

				if done:
					track = {"states": states, "actions": actions, "rewards": rewards, "action_distributions": action_dist}
					trajectories.append(track)
					break

		# parse trajectories
		states = util.flatten([track["states"] for track in trajectories])
		# discount_rewards = self.flatten([self.discount(track["rewards"]) for track in trajectories])
		# print(discount_rewards)
		discount_rewards = []
		for track in trajectories:
			discount_rewards.extend(util.get_discount_reward(track["rewards"], self.gamma))
		discount_rewards = np.asarray(discount_rewards)
		total_reward = sum(util.flatten([track["rewards"] for track in trajectories])) / self.max_episode_num  # average rewards through episodes
		actions = util.flatten([track["actions"] for track in trajectories])
		action_dist = util.flatten([track["action_distributions"] for track in trajectories])

		entropy = entropy / len(actions)

		return states, discount_rewards, total_reward, actions, action_dist, entropy

	##########################################################
	# Step 2: 
	# compute hessian vector product and constraints (kl divergence)
	##########################################################
	def get_mean_kl_divergence(self, model):
		'''
		calculate the mean kl divergence between a given model and new model
		'''
		state_tensors = torch.cat([Variable(torch.from_numpy(state).float().unsqueeze(0)) for state in self.states])
		new_act_probs = model(state_tensors).detach() + 1e-8 # row vector
		old_act_probs = self.pg_model(state_tensors)

		return torch.sum(old_act_probs * torch.log(old_act_probs / new_act_probs), dim = 1).mean(), old_act_probs, new_act_probs

	def fisher_vector_product(self, y):
		'''
		Use fisher vector product J^T M J y

		Args:
		y (torch.Tensor): size 1 * k, k is the length of theta

		Return:
		torch.tensor
		'''

		y= y.view(y.size()[0], -1)
		self.pg_model.zero_grad()
		_, mu_old, mu = self.get_mean_kl_divergence(self.pg_model)

		# mu_old, mu, row tensor

		t = Variable(torch.ones(mu.size()), requires_grad = True)
		fvp = torch.zeros_like(y)

		for i in range(mu.size(0)):
			mu_sample = mu[i]
			J = []
			for mu_s in mu_sample:
				J_ele = torch.autograd.grad(mu_s, self.pg_model.parameters(), create_graph = True)
				J_cat = torch.cat([grad.view(-1) for grad in J_ele])
				J.append(J_cat.view(-1, J_cat.size(0)))
			J = torch.cat(J, dim = 0)

			mu_old_sample = mu_old[i]
			M = torch.zeros((mu.size(1), mu.size(1)))
			for i, mu_old_s in enumerate(mu_old_sample):
				for j, mu_s in enumerate(mu_sample):
					M[i, j] = mu_old_s / (mu_s ** 2)

			fvp += torch.t(J).mm(M).mm(J).mm(y)

		return (fvp / mu.size(0) + self.damping * y).data.view(-1)
	
	def hessian_vector_product(self, y):
		'''
		hessian vector product 
		'''
		self.pg_model.zero_grad()
		mean_kl, _, _ = self.get_mean_kl_divergence(self.pg_model)
		kl_grad = torch.autograd.grad(mean_kl, self.pg_model.parameters(), create_graph = True)
		# to row vector
		kl_grad_row = torch.cat([grad.view(-1) for grad in kl_grad])
		kl_grad_y = (kl_grad_row * y).sum()
		hessian = torch.autograd.grad(kl_grad_y, self.pg_model.parameters())
		# to row vector
		hvp = torch.cat([grad.contiguous().view(-1) for grad in hessian]).data
		return hvp + self.damping * y.data

	##########################################################
	# Step 3: 
	# compute objective (loss) and line search to update
	##########################################################
	def surrogate_loss(self, theta):
		'''
		get surrogate loss

		Args:
		theta (vector): new parameters
		'''
		new_model = deepcopy(self.pg_model)
		vector_to_parameters(theta, new_model.parameters())

		state_tensors = torch.cat([Variable(torch.Tensor(state).float().unsqueeze(0)) for state in self.states])

		prob_new = new_model(state_tensors).gather(1, torch.cat(self.actions).unsqueeze(1)).data
		prob_old = self.pg_model(state_tensors).gather(1, torch.cat(self.actions).unsqueeze(1)).data + 1e-8

		return - torch.mean((prob_new / prob_old) * self.advantage)

	def line_search(self, theta, betas, expected_improve_rate):

		accept_ratio = 0.1
		max_backtrack = 10
		old_loss = self.surrogate_loss(theta)

		for nback, shrink in enumerate(0.5 ** np.arange(max_backtrack)):
			theta_new = theta.data.numpy() + shrink * betas

			theta_new_var = Variable(torch.from_numpy(theta_new).float())
			new_loss = self.surrogate_loss(theta_new_var)

			diff = old_loss - new_loss
			expected = expected_improve_rate * shrink
			ratio = diff / expected

			if ratio > accept_ratio and diff > 0:
				return theta_new_var
		return theta

	def line_search_v2(self, theta):
		'''
		line search to return the parameter vector
		'''
		old_loss = self.surrogate_loss(theta)
		old_loss = Variable(old_loss, requires_grad = True)
		params = torch.cat([param.view(-1) for param in self.pg_model.parameters()])	
		old_loss.backward(params)
		old_loss_grad = old_loss.grad
		s = self.conjugate_gradient(old_loss_grad)

		s = torch.from_numpy(s).float()
		beta = torch.sqrt(2 * self.delta / (s * old_loss_grad).sum())
		
		beta_end = 0
		decay = 100
		alpha = 0.1
		for d in range(decay):
			beta = beta * math.exp(- alpha * d) # shrink exponentially
			theta_new = theta + beta * s

			# compute objective
			new_loss = self.surrogate_loss(theta_new)

			new_model = deepcopy(self.pg_model)
			vector_to_parameters(theta_new, new_model.parameters())

			mean_kl, _, _ = self.get_mean_kl_divergence(new_model)

			if mean_kl <= self.delta and new_loss < old_loss: # objective improve
				return theta_new
		return theta


	def conjugate_gradient(self, b):
		'''
		Conjugate gradient method 
		
		Reference:
		https://en.wikipedia.org/wiki/Conjugate_gradient_method
		https://github.com/mjacar/pytorch-trpo/blob/master/trpo_agent.py

		Args:
		b (torch.Variable)

		Return:
		x (numpy array)

		Note:
		all use float tensor
		torch dot operation only accepts 1 dim
		'''
		threshold = 1e-10
		p = b.clone().data.float() # to tensor
		r = b.clone().data.float()

		x = np.zeros_like(b.data.numpy(), dtype = np.float64)
		k = 0

		while k < self.cg_iter:
			rdot = r.dot(r)
			hvp = self.hessian_vector_product(Variable(p)).float().squeeze(0)

			alpha = rdot / p.dot(hvp)
			x += alpha.numpy() * p.numpy()
			r -= alpha * hvp

			newrdot = r.dot(r)
			beta = newrdot / rdot
			p = r + beta * p
			if torch.sqrt(newrdot) < threshold:
				break

			k += 1
		return x

	##########################################################
	# Step 4: 
	# step to update
	##########################################################

	def step(self, verbose = 1):

		# sample trajectory
		states, discount_rewards, total_reward, actions, action_dist, entropy = self.sample_trajectory()

		# batch 
		# num_batch = len(actions) // self.batch_size if len(actions) % self.batch_size == 0 else len(actions) // self.batch_size + 1
		num_batch = 1

		# loop in batches
		for bid in range(num_batch):
			# print("Start to process batch id {}...".format(bid))

			# sample batches for rgb training
			#################################
			start_id = bid * self.batch_size
			end_id = (bid + 1) * self.batch_size

			start_id = 0
			end_id = len(states)
			self.states = states[start_id:end_id]
			self.discount_rewards = discount_rewards[start_id:end_id]
			self.actions = actions[start_id:end_id]
			self.action_dist = action_dist[start_id:end_id]

			# calculate the advantage
			V = self.val_model.predict(self.states).data
			Q = torch.Tensor(self.discount_rewards).unsqueeze(1)
			advantage = Q - V

			# normalize the advantage
			self.advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

			# initialize surrogate loss
			num_actions = self.env.action_space.n
			new_p = torch.cat(self.action_dist).view(-1, num_actions).gather(1, torch.cat(self.actions).unsqueeze(1))
			old_p = new_p.detach() + 1e-8

			surrogate_loss = - torch.mean(new_p / old_p * Variable(self.advantage)) - self.lamb * entropy # self.lamb * self.entropy regularazation term

			# calculate policy gradient
			self.pg_model.zero_grad()
			surrogate_loss.backward(retain_graph = True)
			policy_gradient = parameters_to_vector([v.grad for v in self.pg_model.parameters()]).squeeze(0)

			# loop if has gradient
			nozero_grad = policy_gradient.nonzero().size()[0]

			if nozero_grad:

				# move direction d
				d = self.conjugate_gradient(- policy_gradient) # loss and gradient positive and negative
				d_var = Variable(torch.from_numpy(d).float())

				# line search
				sTAs = 0.5 * d.dot(self.hessian_vector_product(d_var).numpy().T)
				lm = np.sqrt(sTAs / self.delta)
				betas = d / lm

				expected_improvate_rate = - policy_gradient.dot(d_var).item() / lm
				theta_old = parameters_to_vector(self.pg_model.parameters())
				theta = self.line_search(theta_old, betas, expected_improvate_rate)

				# update value function
				error = util.explained_variance_1d(V.squeeze(1).numpy(), self.discount_rewards)

				self.val_model.fit(self.states, Variable(torch.Tensor(self.discount_rewards)))
				value_model_params = parameters_to_vector(self.val_model.parameters())

				new_V = self.val_model.predict(self.states).data.squeeze(1).numpy()
				error_new = util.explained_variance_1d(new_V, self.discount_rewards)

				if error_new < error or np.abs(error_new) < 1e-4:
					# update value model
					vector_to_parameters(value_model_params, self.val_model.parameters())

				# update policy model  
				old_model = deepcopy(self.pg_model)
				old_model.load_state_dict(self.pg_model.state_dict())
				if any(np.isnan(theta.data.numpy())):
					print("No update")
				else:
					vector_to_parameters(theta, self.pg_model.parameters())

				if verbose:
					old_kl, _, _ = self.get_mean_kl_divergence(old_model)
					info = collections.OrderedDict([("Total reward", total_reward), ("Old kl", old_kl.item()),
						("Error Value before", error), ("Error Value after", error_new)])
					for key, value in info.items():
						print("{}: {}".format(key, value))

			else:
				print("Policy gradient is 0. No update!")
		return total_reward


if __name__ == '__main__':
	env = gym.make("CartPole-v0").unwrapped
	numActions = env.action_space.n
	pg = model.PolicyNet(numActions)
	value = model.ValueFunctionWrapper(model.ValueNet(), lr = 0.01)
	env = gym.make("CartPole-v0").unwrapped
	max_iter = 300

	trpo = TRPO(pg, value, env)
	rewards = []
	bestReward = float("-inf")
	bestModel = None
	for it in range(max_iter):
		
		reward = trpo.step(verbose = 0)
		rewards.append(reward)

		if it % 100 == 0:
			print("Iteration {}...".format(it))
			print("reward: ", reward)

		if reward > bestReward:
			bestReward = reward
			bestModel = deepcopy(trpo.pg_model)

	plt.plot(list(range(max_iter)), rewards, "r-")
	plt.xlabel("episodes")
	plt.ylabel("reward")
	plt.show()

	state = env.reset()
	done = False

	reward_sum = 0
	while not done:
		state = torch.from_numpy(state).float().unsqueeze(0)
		action_prob = bestModel(state)
		action = torch.argmax(action_prob).item()

		newState, reward, done, _ = env.step(action)
		reward_sum += reward
		env.render()

		state = newState
	env.close()
	print("total reward: ", reward_sum)
