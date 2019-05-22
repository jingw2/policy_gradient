#!/usr/bin/python 3.6
# -*-coding:utf-8-*-

'''
Network function approximator
Author: Jing Wang (jingw2@foxmail.com)
'''

# torch
from torch.autograd import Variable
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self, state_size, hidden_size, output_size, n_layers,
            activation=nn.ReLU(), output_activation=None, discrete=True,
            min_logstd=-0.5, max_logstd=0.5):
        '''
        Build feedforward neural network
        '''
        super(Net, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.n_layers = n_layers
        self.discrete = discrete
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

        layers = []
        for i in range(self.n_layers):
            if i == 0:
                layer = nn.Linear(state_size, hidden_size)
            else:
                layer = nn.Linear(hidden_size, hidden_size)
            layers.extend([layer, self.activation])
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, output_size)

        # if continuous actions, the log standard deviation of actions in GP
        if not self.discrete:
            self.logstd_layer = nn.Linear(hidden_size, 1)    
    
    def forward(self, x):
        x = self.layers(x)
        out = self.output(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        if not self.discrete:
            logstd = self.logstd_layer(x)
            logstd = torch.clamp(logstd, self.min_logstd, self.max_logstd)
            return out, logstd
        else:
            return out

class Actor(nn.Module):

    def __init__(self, state_size, hidden_size, output_size, n_layers,
            activation=nn.ReLU(), output_activation=None):

        super(Actor, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.n_layers = n_layers

        layers = []
        for i in range(self.n_layers):
            if i == 0:
                layer = nn.Linear(state_size, hidden_size)
            else:
                layer = nn.Linear(hidden_size, hidden_size)
            layers.extend([layer, self.activation])
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layers(x)
        out = self.output(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out

class QNet(nn.Module):

    def __init__(self, state_size, hidden_size, output_size, n_layers, action_size,
            activation=nn.ReLU(), output_activation=None):
        super(QNet, self).__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.n_layers = n_layers

        layers = []
        for i in range(self.n_layers):
            layer = nn.Linear(hidden_size, hidden_size)
            layers.extend([layer, self.activation])
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, output_size)
        self.combine_layer = nn.Linear(state_size + action_size, hidden_size)

    def forward(self, x, ac):
        '''
        x (tensor): state (batch_size, state_size)
        ac (tensor): action (batch_size, action_size)
        '''
        x = self.activation(self.combine_layer(torch.cat([x, 
            ac], dim=1)))
        x = self.layers(x)
        out = self.output(x)
        if self.output_activation is not None:
            out = self.output_activation(out, dim=1)
        return out

