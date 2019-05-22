#!/usr/bin/python 3.6
# -*-coding:utf-8-*-

'''
Soft Actor Critic
Author: Jing Wang
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
from torch.nn.utils.convert_parameters import \
    vector_to_parameters, parameters_to_vector

class SAC(agent.Agent):

    def __init__(self, net_args, trajectory_args, reward_args, sac_args):
        '''
        Args:
        the first tree arguments see agent.Agent

        sac_args:
            - tau (float): soft update coefficient
            - automated_entropy_tuning (bool): use automated entropy tuning (recommended)
            - duel_q_net (bool): use duel q network to speed up training (recommended)
            - action_bound_fn (str): use tanh or sigmoid to enforce action bound
            - policy_type (str): use gaussian policy or deterministic policy
        '''
        super(SAC, self).__init__(net_args, trajectory_args, reward_args)
        self.tau = sac_args["tau"]
        self.duel_q_net = sac_args["duel_q_net"]
        self.policy_type = sac_args["policy_type"]
        self.action_bound_fn = sac_args["action_bound_fn"]

    def build_net(self):
        '''
        build network based on parameters input

        Actor Network:
         * Use gaussian:
            the network outputs mean and log standard deviation, 
            so the output size should be action size
         * Use determinstic:
            the network only outputs mean with action size, 
            so set the discrete as True that log standard deviation is None 
        
        Value Network:
         * output the estimated value based on replay buffer

        Q network:
         * output the Q value

        Target Network:
         * use to update Q network
        '''

        # build net
        if self.policy_type == "gaussian":
            self.actor = model.Net(self.state_size, self.hidden_size, self.action_size, \
                self.n_layers, output_activation=self.output_activation, discrete=self.discrete)
            # automated entropy adjustment for maximum entropy rl
            self.entropy_target = - self.action_size
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
            # target q network
            self.target_q_net = model.QNet(self.state_size, self.hidden_size, 1, \
                self.n_layers, self.action_size, output_activation=None)

        elif self.policy_type == "deterministic":
            self.actor = model.Net(self.state_size, self.hidden_size, self.action_size, \
                self.n_layers, output_activation=self.output_activation, discrete=self.discrete)
            self.value_net = model.Actor(self.state_size, self.hidden_size, 1, \
                self.n_layers, output_activation=None)
            self.target_value_net = deepcopy(self.value_net)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        
        if self.duel_q_net:
            # duel Qnet
            self.q1_net = model.QNet(self.state_size, self.hidden_size, 1, \
                self.n_layers, self.action_size, output_activation=None)
            self.q2_net = deepcopy(self.q1_net)
        else:
            self.q_net = model.QNet(self.state_size, self.hidden_size, 1, \
                self.n_layers, self.action_size, output_activation=None)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        
        if self.duel_q_net:
            self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=self.lr)
            self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=self.lr)
        else:
            self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # loss function
        self.value_loss_fn = nn.MSELoss()
        if self.duel_q_net:
            self.q1_loss_fn = nn.MSELoss()
            self.q2_loss_fn = nn.MSELoss()
        else:
            self.q_loss_fn = nn.MSELoss()

        # replay memory
        self.replay_memory = util.ReplayMemory(1000000)
        self.transition = collections.namedtuple("transition", ["state", 
                "action", "next_state", "reward"])
        
        # entropy target
        self.entropy_target = - self.action_size
        self.noise = OUNoise(mu=np.zeros(self.action_size), sigma=0.1)
        self.epsilon = 1.
    
    def sample_actions(self, states_input, ounoise=True, decay=True):
        '''
        Sample actions based on states input

        Gaussian Policy:
            z ~ N(mu, sigma), mu and sigma are generated from network, 
                reparameterization trick for lower variance estimator
        Deterministic:
            z ~ mu + noise, mu is generated from network
        Enforce action bound:
            - enforce (-1, 1): action = tanh(z)
            - enforce (0, 1): action = sigmoid(z)
        '''
        mean, logstd = self.actor(states_input)
        dist = torch.distributions.Normal(mean, torch.exp(logstd))
        z = dist.sample()
        if self.action_bound_fn == "tanh":
            sampled_actions = torch.tanh(z).detach().cpu().numpy()
        elif self.action_bound_fn == "sigmoid":
            sampled_actions = torch.sigmoid(z).detach().cpu().numpy()
        return sampled_actions[0]
    
    def get_log_prob(self, states_input):
        '''
        Since
            pi(a|s) = mu(u|s)|det(da / du)^(-1)|
        da / du is a diagonal matrix, det(da/du) = diagonal(da / du)

        Tanh:
            da /du = 1 - tanh^2 u = 1 - a^2
            log (pi | s) = log mu(u|s) - sum_i (log (1 - a_i^2))
        Sigmoid:
            da / du = - 1 / (1 + e^(-u)) + 1 / (1 + e^(-u))^2
                    = - a + a^2
            log (pi | s) = log mu(u|s) - sum_i (log (a_i^2 - a_i))
        In the end, take the sume along dim=1
        '''
        mean, logstd = self.actor(states_input)
        dist = torch.distributions.Normal(mean, torch.exp(logstd))
        z = dist.sample()
        # enforcing action bounds (-1, 1)
        if self.action_bound_fn == "tanh":
            action = torch.tanh(z)
            da_du = 1 - action.pow(2)
        elif self.action_bound_fn == "sigmoid":
        # enforcing action bounds (0, 1)
            action = torch.sigmoid(z)
            da_du = - action + action.pow(2)
        logprob = dist.log_prob(z) - torch.log(da_du + 1e-8)
        logprob = logprob.sum(1, keepdim=True)
        return action, logprob, z, mean, logstd
    
    def train_op(self):
        '''
        Steps:
         - Sample batch and combine data

         Deterministic:
            - Target value network input s_{t+1} from replay buffer: 
                * V_{psi_bar}(s_{t+1})
            - Q network (replay buffer data): (if duel Q network, use the minimal one)
                * Q_{hat}(s_t, a_t) = r(s_t, a_t) + gamma * E (V_{psi_bar}(s_{t+1}))    
                * Q(s_t, a_t)
            - Value network: 
                (Note that actions here are sampled according to current policy and states)
                * V_{psi}(s_t)
            - Q network and log probability, a' is sampled 
                * Q(s_t, a'_t) - log pi(a'_t|s_t)
            - loss function:
                * value loss:
                    J(psi) = MSELoss (V_{psi}(s_t) - mean(Q(s_t, a'_t) - log pi(a'_t|s_t)))
                * q loss:
                    J(theta) = MSELoss (Q(s_t, a_t) - Q_{hat}(s_t, a_t))
                * policy loss:
                    J(phi) = log pi(a'_t|s_t) - Q(s_t, a'_t) 
        Gaussian:
            - Sample actions to get next tate actions, a'_{t+1} sampled based on s_{t+1}
            - Target Q network:
                V = r(s_t, a_t) + gamma * Q_{theta_bar}(s_{t+1}, a'_{t+1}) - alpha * log pi(a'_{t+1}|s_{t+1})
            - Q network
                Q_{theta} (s_t, a_t)
            - Q loss: 
                J(theta) = MSELoss (Q_{theta}(s_t, a_t) - V)
            - Actor loss:
                J(phi) = alpha * log pi(a'_t|s_t) - Q_{theta} (s_t, a'_t)
            - alpha loss:
                J(alpha) = -alpha * log pi(a_t|s_t) - alpha * entropy_target

        '''
        if len(self.replay_memory) < self.min_timesteps_per_batch:
            return 
        
        batch = self.replay_memory.sample(self.min_timesteps_per_batch)
        
        not_end_index = torch.tensor(tuple(map(lambda b: 
            b.next_state is not None, batch)), dtype=torch.uint8)

        # batch data
        next_state_batch = torch.cat([torch.Tensor(b.next_state).float() for b in batch
            if b.next_state is not None]).view(-1, self.state_size)
        state_batch= torch.cat([torch.Tensor(b.state).float() 
            for b in batch]).view(-1, self.state_size)
        action_batch = torch.cat([torch.Tensor(b.action).float() 
            for b in batch]).view(-1, self.action_size)
        reward_batch = torch.cat([torch.Tensor([b.reward]).float() 
            for b in batch]).view(-1)
        
        if self.policy_type == "deterministic":
            target_value = torch.zeros(self.min_timesteps_per_batch).float()
            target_value[not_end_index] = self.target_value_net(next_state_batch).view(-1)
            next_qvalue = reward_batch + self.gamma * target_value
        elif self.policy_type == "gaussian":
            target_qvalue = torch.zeros(self.min_timesteps_per_batch).float()
            sample_next_action, next_logprob, _, _, _ = self.get_log_prob(next_state_batch)
            target_qvalue[not_end_index] = self.target_q_net(next_state_batch, sample_next_action)
            next_qvalue = self.gamma * (target_qvalue - torch.exp(self.log_alpha) \
                * next_logprob) + reward_batch

        q1_pred = self.q1_net(state_batch, action_batch).view(-1)
        q2_pred = self.q2_net(state_batch, action_batch).view(-1)
        sample_action, logprob, z, mean, logstd = self.get_log_prob(state_batch)

        # here for value net, actions are sampled from current policy, not replay buffer
        if self.duel_q_net:
            new_q1_pred = self.q1_net(state_batch, sample_action)
            new_q2_pred = self.q2_net(state_batch, sample_action)
            new_q_pred = torch.min(new_q1_pred, new_q2_pred)
        else:
            new_pred = self.q_net(state_batch, sample_action)

        # loss 
        if self.policy_type == "gaussian":
            actor_loss = (self.alpha * logprob - new_q_pred).mean()
        elif self.policy_type == "deterministic":
            actor_loss = (logprob - new_q_pred).mean()
            actor_loss = actor_loss.mean()
            # value network
            next_value = new_q_pred - logprob
            value_pred = self.value_net(state_batch)
            value_loss = self.value_loss_fn(value_pred, next_value.detach()).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
      
        if self.duel_q_net:
            q1_loss = self.q1_loss_fn(q1_pred, next_qvalue.detach()).mean()
            q2_loss = self.q2_loss_fn(q2_pred, next_qvalue.detach()).mean()
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.q1_net.parameters(), 0.5)
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.q2_net.parameters(), 0.5)
            self.q2_optimizer.step()
        else:
            q_loss = self.q2_loss_fn(q_pred, next_qvalue.detach()).mean()
            self.q_optimizer.zero_grad()
            q_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
            self.q_optimizer.step()

        if self.policy_type == "deterministic":
            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)
        elif self.policy_type == "gaussian":
            if self.duel_q_net:
                for target_param, param1, param2 in zip(self.target_value_net.parameters(), self.q1_net.parameters(), 
                        self.q2_net.parameters):
                    param = random.choice([param1, param2])
                    target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)
            else:
                for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                    target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)
            
            # automated entropy tuning
            alpha = torch.exp(self.log_alpha) 
            alpha_loss = - (alpha * (logprob + self.entropy_target).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
    
    def sample_trajectory(self, env):
        state = env.reset()
        #TODO:
        # env.reset()
        # state = env.state
        states, actions, rewards = [], [], []
        steps = 0
        while True:
            #TODO:
            # if np.isnan(state):
            #     env.reset()
            #     state = env.state
            #     continue
            # state = np.array([state])
            states.append(state)
            state = torch.Tensor(state.reshape(1, -1)).float()
            action = self.sample_actions(state)
            
            #TODO:
            action = np.clip(action, env.action_space.low, env.action_space.high).tolist()
            # action = np.clip(action, 1e-7, 1).tolist()
            if np.isnan(action[0]):
                env.reset()
                state = env.state
                continue
            actions.append(action)

            #TODO:
            next_state, reward, done, _ = env.step(action)
            # next_state, reward, done, _ = env.evaluateAction(action)
            if done:
                next_state = None
                self.replay_memory.push(self.transition(state, action, 
                    None, reward))
            else:
                self.replay_memory.push(self.transition(state, action, 
                    np.array([next_state]), reward))
            
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
