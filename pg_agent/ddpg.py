#!/usr/bin/python 3.6
# -*-coding:utf-8-*-

'''
Deep Deterministic Policy Gradient
Author: Jing Wang (jingw2@foxmail.com)
'''

#sys
import sys
sys.path.append('../')
import numpy as np
import time
from multiprocessing import Process
import os
import inspect
import random
from copy import deepcopy
import collections

# gym
import gym

# self-defined
import model
import agent
import utils.logz as longz
import utils.util as util
from utils.util import OrnsteinUhlenbeckActionNoise as OUNoise

# pytorch
from torch.autograd import Variable
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from netsapi.challenge import *

class DDPG(agent.Agent):

    def __init__(self, net_args, trajectory_args, reward_args, ddpg_args):
        '''
        Args:
        the first tree arguments see agent.Agent

        ddpg_args:
            - tau (float): soft update coefficient
            - ounoise (bool): use ounoise to explore action (recommended)
            - decay (bool): use decay to sample actions (recommended)
        '''
        super(DDPG, self).__init__(net_args, trajectory_args, reward_args)
        self.tau = ddpg_args["tau"]
        self.ounoise = ddpg_args["ounoise"]
        self.decay = ddpg_args["decay"]

    def build_net(self):
        '''build actor, critic, target_actor, target_critic network'''

        # actor
        self.actor = model.Actor(self.state_size, self.hidden_size, self.action_size, \
            self.n_layers, output_activation=self.output_activation)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)

        # critic
        self.critic = model.QNet(self.state_size, self.hidden_size, 1, \
            self.n_layers, self.action_size, output_activation=None)
        self.target_critic = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # replay memory
        self.replay_memory = util.ReplayMemory(1000000)
        self.transition = collections.namedtuple("transition", ["state", 
                "action", "next_state", "reward"])
        
        if self.ounoise:
            self.noise = OUNoise(mu=np.zeros(self.action_size), sigma=0.1)
        self.epsilon = 1.
    
    def sample_actions(self, states_input):
        '''
        mean is generated from actor network
        action = mean + noise
        '''
        if self.decay:
            self.epsilon -= 1 / 50000
        action = self.actor(states_input)
        if self.ounoise:
            with torch.no_grad():
                noise = self.epsilon * Variable(torch.FloatTensor(self.noise()))
        else:
            noise = np.random.normal(0, 0.2, size=self.action_size)
        
        noise = torch.Tensor(noise).float()
        sampled_actions = action + noise
        # sampled_actions = mean + torch.exp(logstd) * torch.randn(mean.size())
        sampled_actions = sampled_actions.data.numpy()[0]
        return sampled_actions

    def train_op(self):
        '''
        Steps:
            - a_{t+1} is sampled from target actor based on s_{t+1}
            - Q_prime (s_{t+1}, a_{t+1}) is from target critic
            - Qtrue (s_t, a_t) = r(s_t, a_t) + gamma * Q_prime (s_{t+1}, a_{t+1})
            - Qpred (s_t, a_t) is sampled from critic based on s_t, a_t
            - Critic loss: MSELoss (Qtrue, Qpred)
            - Actor loss: - Mean [Q(s_t, a'_t)], a'_t is sampled from current actor
        '''
        if len(self.replay_memory) < self.min_timesteps_per_batch:
            return 
        
        batch = self.replay_memory.sample(self.min_timesteps_per_batch)
        
        not_end_index = torch.tensor(tuple(map(lambda s: 
            s.next_state is not None, batch)), 
            dtype=torch.uint8)

        next_state_batch = torch.cat([torch.Tensor(b.next_state).float() for b in batch
            if b.next_state is not None]).view(-1, self.state_size)

        state_batch= torch.cat([torch.Tensor(b.state).float() 
            for b in batch]).view(-1, self.state_size)
        action_batch = torch.cat([torch.Tensor(b.action).float() 
            for b in batch]).view(-1, self.action_size)
        reward_batch = torch.cat([torch.Tensor([b.reward]).float() 
            for b in batch]).view(-1)

        # target Q-value
        Q_prime = torch.zeros(self.min_timesteps_per_batch).float()
        next_action_batch = self.target_actor(next_state_batch)
        Q_prime[not_end_index] = self.target_critic(next_state_batch, 
            next_action_batch).view(-1)
        
        y = reward_batch + self.gamma * Q_prime.detach()

        # critic Q-value
        Q = self.critic(state_batch, action_batch).view(-1)

        # critic loss and update
        mse = nn.MSELoss()
        critic_loss = mse(Q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor loss
        actor_loss = - self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update targets
        self.target_critic = self.update_target(self.critic, self.target_critic)
        self.target_actor = self.update_target(self.actor, self.target_actor)

    def update_target(self, source, target):
        '''soft update target'''
        new_target_param = parameters_to_vector(source.parameters()) * self.tau + \
                (1 - self.tau) * parameters_to_vector(target.parameters())
        vector_to_parameters(new_target_param, target.parameters())
        return target
    
    def sample_trajectory(self, env):
        # state = env.reset()
        #TODO:
        env.reset()
        state = env.state
        states, actions, rewards = [], [], []
        steps = 0
        while True:
            #TODO:
            state = np.array([state])
            states.append(state)
            state = torch.Tensor(state.reshape(1, -1)).float()
            action = self.sample_actions(state)
            
            #TODO:
            # action = np.clip(action, env.action_space.low, env.action_space.high).tolist()
            action = np.clip(action, 1e-7, 1).tolist()
            actions.append(action)

            #TODO:
            # next_state, reward, done, _ = env.step(action)
            next_state, reward, done, _ = env.evaluateAction(action)
            if not done:
                self.replay_memory.push(self.transition(state, action, 
                    np.array([next_state]), reward))
            else:
                self.replay_memory.push(self.transition(state, action, 
                    None, reward))

            
            state = next_state
            rewards.append(reward)
            steps += 1

            if done or steps > self.max_path_length:
                break
        
        path = {
                "state": np.array(states, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
                "action": np.array(actions, dtype=np.float32)
                }
        return path
    
    def set_params_noise(self):
        '''set parameter noise'''
        raise NotImplementedError
