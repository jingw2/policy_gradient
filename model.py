# /usr/bin/env python 3.6
# -*-coding:utf-8-*-


'''
Policy and Value Network Model
Author: Jing Wang (jingw2@foxmail.com)

Reference link:
https://github.com/mjacar/pytorch-trpo/blob/master/utils/torch_utils.py
'''

import torch
from torch.autograd import Variable
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

class PolicyNet(nn.Module):
	def __init__(self, numActions):
		super(PolicyNet, self).__init__()
		self.layer1 = nn.Linear(4, 10)
		self.layer2 = nn.Linear(10, numActions)
		self.softmax = nn.Softmax()
		
	
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = self.layer2(x)
		x = F.softmax(x, dim = 1)
		# x = x.view(x.size(1),  -1)
		return x

class ValueNet(nn.Module):

	def __init__(self):
		super(ValueNet, self).__init__()
		self.layer1 = nn.Linear(4, 10)
		self.layer2 = nn.Linear(10, 1)

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = self.layer2(x)

		return x

class ValueFunctionWrapper(nn.Module):
	"""
	Wrapper around any value function model to add fit and predict functions
	"""

	def __init__(self, model, lr):
		super(ValueFunctionWrapper, self).__init__()
		self.model = model
		self.loss_fn = nn.MSELoss()
		self.lr = lr

	def forward(self, data):
		return self.model.forward(data)

	def fit(self, observations, labels):
		def closure():
			predicted = self.predict(observations)
			predicted = predicted.view(-1)
			loss = self.loss_fn(predicted, labels)
			self.optimizer.zero_grad()
			loss.backward()
			return loss
		old_params = parameters_to_vector(self.model.parameters())
		for lr in self.lr * .5**np.arange(10):
			self.optimizer = optim.LBFGS(self.model.parameters(), lr=lr)
			self.optimizer.step(closure)
			current_params = parameters_to_vector(self.model.parameters())
		if any(np.isnan(current_params.data.cpu().numpy())):
			print("LBFGS optimization diverged. Rolling back update...")
			vector_to_parameters(old_params, self.model.parameters())
		else:
			return

	def predict(self, observations):
		return self.forward(torch.cat([Variable(torch.Tensor(observation)).unsqueeze(0) for observation in observations]))