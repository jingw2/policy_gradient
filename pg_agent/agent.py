#!/usr/bin/python 3.6
# -*-coding:utf-8-*-

'''
Agent object, modified from hw2 of cs294-112
Author: Jing Wang (jingw2@foxmail.com)
'''

#sys
import sys
sys.path.append('../')
import numpy as np
import time
from multiprocessing import Process
import os
import utils.logz as logz
import inspect
import random
import pandas as pd

from netsapi.challenge import *

# gym
import gym

# self-defined
import model

# pytorch
from torch.autograd import Variable
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Agent(object):

    def __init__(self, net_args, trajectory_args, reward_args):
        '''
        Args:

        net_args (dict): network arguments
            - state_size
            - action_size
            - discrete: boolean to check if the action is continuous or discrete
            - hidden_size
            - n_layers
            - learing_rate
        
        trajectory_args (dict): sample trajectories arguments
            - max_path_length: limit the length of path
            - min_timesteps_per_batch
        
        reward_args (dict): reward related arguments
            - gamma: discount factor
            - nn_baseline: boolean to initiate baseline method
            - normalize_advantage
        '''
        super(Agent, self).__init__()
        self.state_size = net_args["state_size"]
        self.action_size = net_args["action_size"]
        self.discrete = net_args["discrete"]
        self.hidden_size = net_args["hidden_size"]
        self.n_layers = net_args["n_layers"]
        self.output_activation = net_args["output_activation"]
        self.lr = net_args["learing_rate"]

        self.max_path_length = trajectory_args["max_path_length"]
        self.min_timesteps_per_batch = trajectory_args["min_timesteps_per_batch"]

        self.gamma = reward_args["gamma"]
        self.nn_baseline = reward_args["nn_baseline"]
        self.normalize_advantage = reward_args["normalize_advantage"]
    
    def build_net(self):
        '''build network based on arguments input'''
        self.net = model.Net(self.state_size, self.hidden_size, self.action_size, \
            self.n_layers, output_activation=self.output_activation, discrete=self.discrete)
        self.net_optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
        if self.nn_baseline:
            self.critic = model.Net(self.state_size, self.hidden_size, 1, self.n_layers, \
                output_activation=None, discrete=self.discrete)
            self.base_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
    
    def policy_forward(self, states_input):
        '''
        Use network to return parameters of pi(a|s)

        Args:
        states_input (tensor): shape is (batch_size, state_size)

        Returns:
        the parameter of policy
            - if discrete actions, the parameters are the logits of 
                a categorical distribution over the actions
              * logits (tensor): (batch_size, action_size)
            - if continuous actions, the parameters are the mean and 
                log std of Gaussian distribution of the actions
              * mean (tensor): (batch_size, action_size)
              * logstd (tensor): (action_size,)
        '''

        if self.discrete:
            logits = self.net(states_input)
            return logits
        else:
            mean, logstd = self.net(states_input)
            return (mean, logstd)
    
    def sample_actions(self, policy_params):
        '''
        Select actions based on policy parameters from policy forward

        Args:
        states_input (tensor): shape is (batch_size, state_size)

        Return:
        if discrete:
            * return action with shape (batch_size,)
        else:
            * return action with shape (batch_size, action_size)
        
        Note:
            for continuous actions, actions are sampled from
                mu + sigma * z, z ~ N(0, 1)
        '''
        if self.discrete:
            logits = policy_params
            prob = torch.distributions.Categorical(logits=logits)
            sampled_actions = prob.sample()
        else:
            mean, logstd = policy_params
            sampled_actions = mean + torch.exp(logstd) * torch.randn(mean.size())
        return sampled_actions
    
    def get_log_prob(self, policy_params, actions_input):
        '''
        Get log probability based on policy parameters and actual actions

        Args:
        policy_params (See output from policy_forward function)
        actions_input (tensor): (batch_size,) for discrete, 
                                (batch_size, action_size) for continuous
        
        Returns:
        logprob (tensor): (batch_size,)

        Note:
        for discrete actions, the log probability can be calculate from cross entropy loss, 
            which is
                log pi(a|s) = - CEH(y, pi(a|s))
                CEH(y, pi(a|s)) = - [y log pi(a|s) + (1 - y) log (1 - pi(a|s))]
            where y is the 0-1 actual action selection
        for continous actions, the log probability can be calculate by MSE loss alike
            which is 
                log pi(a|s) = 0.5 * sum ((mean - actions_input) / std)**2
        '''

        if self.discrete:
            logits = policy_params
            cross_entropy_loss = nn.CrossEntropyLoss(reduce=False, reduction=None)
            logprob = - cross_entropy_loss(logits, actions_input.long())
        else:
            mean, logstd = policy_params
            z = (mean - actions_input) / torch.exp(logstd)
            logprob = - 0.5 * torch.sum(torch.mul(z, z), dim=1)
        return logprob

    def train_op(self, states_input, actions_input, q_n, advantage_input):
        '''
        Define train options: loss, optimizer

        Args:
        states_input (tensor): (batch_size, state_size)
        actions_input (tensor): (batch_size,) or (batch_size, action_size)
        advantage_input (tensor): (sum of path lengths), depend on the path sampled
        '''

        self.policy_params = self.policy_forward(states_input)
        self.logprob = self.get_log_prob(self.policy_params, actions_input)
        
        # calculate loss
        loss = - torch.mul(self.logprob, advantage_input).mean()
        
        self.net_optimizer.zero_grad()
        loss.backward()
        self.net_optimizer.step()

        # use baseline or critic
        if self.nn_baseline:
            self.base_pred = self.critic(states_input).view(-1)
            mse = nn.MSELoss()
            target_input = (q_n - q_n.mean()) / (q_n.std() + 1e-8)
            base_loss = mse(self.base_pred, target_input.detach())
        
            self.base_optimizer.zero_grad()
            base_loss.backward()
            self.base_optimizer.step()
    
    def sample_trajectories(self, iters, env):
        '''
        Sample trajectories data

        Args:
        iters (int): iteration
        env : simulation environment

        Returns:
        paths (list): paths data
        timesteps (int): time steps of this batch
        '''

        timesteps = 0
        paths = []
        while True:
            path = self.sample_trajectory(env)
            paths.append(path)
            timesteps += pathlength(path)

            if timesteps > self.min_timesteps_per_batch:
                break
        return paths, timesteps
    
    def sample_trajectory(self, env):
        state = env.reset()
        #TODO:
        env.reset()
        state = env.state
        states, actions, rewards, next_states = [], [], [], []
        steps = 0
        while True:
            #TODO:
            state = np.array([state])
            states.append(state)
            state = torch.Tensor(state.reshape(1, -1)).float()
            policy_params = self.policy_forward(state)
            action = self.sample_actions(policy_params)
            action = action.data.numpy()[0]
            #TODO:
            action = np.clip(action, 1e-7, 1).tolist()
            actions.append(action)

            #TODO:
            # state, reward, done, _ = env.step(action)
            state, reward, done, _ = env.evaluateAction(action)
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
    
    def get_discount_rewards(self, rewards):
        '''
        Monte carlo simulation of the Q-function

        Args:
        rewards (list): each element is an array containing the rewards of paths

        Return:
        q_n (array like): sum of path lengths, a single vector for the estimated 
                          q values
        '''
        sum_of_path_lengths = sum([len(r) for r in rewards])
        q_n = []
        
        for reward in rewards:
            r = 0
            q = [0] * len(reward)
            for i in range(len(reward)-1, -1, -1):
                r = reward[i] + self.gamma * r
                q[i] = r
            q_n.extend(q)
        
        q_n = torch.Tensor(np.asarray(q_n)).float().view(-1)
        return q_n
    
    def compute_advantage(self, states_input, q_n):
        '''
        Compute advantages by subtracting a baseline from the estimated Q values

        Args:
        states_input (tensor): (sum of path lengths, state_size)
        q_n (tensor): (sum of path lengths)

        Return:
        adv (tensor): (sum of path lengths)
        '''
        if self.nn_baseline:
            b_n = self.critic(states_input)
            # normalize b_n based on q_mean and q_std
            b_n = q_n.mean() + (q_n.std() + 1e-8) * ((b_n - b_n.mean()) / (b_n.std() + 1e-8))
            adv = q_n - b_n
        else:
            adv = q_n.clone() 
        return adv
    
    def estimate_return(self, states_input, rewards):
        '''
        Estimate the rewards over a set of trajectories

        Args:
        states_input (tensor): (sum of path lengths, state_size)
        rewards (list): number of paths

        Return:
        q_n (tensor): sum of path lengths
        adv (tensor): sum of path lengths
        '''
        q_n = self.get_discount_rewards(rewards)
        adv = self.compute_advantage(states_input, q_n)

        if self.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return q_n, adv

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)


def train_PG(
    exp_name, 
    env_name, 
    n_iters, 
    gamma, 
    min_timesteps_per_batch, 
    max_path_length, 
    lr, 
    normalize_advantages, 
    nn_baseline,
    seed, 
    n_layers,
    hidden_size,
    discrete,
    logdir):

    start = time.time()

    # env
    env = gym.make(env_name)
    #TODO:
    # env = ChallengeSeqDecEnvironment(experimentCount=3005, userID="jingw2", \
    #     timeout=5, realworkercount=8)
    # env.state_size = 1
    # env.action_size = 2

    # set up logger
    setup_logger(logdir, locals())

    # random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(env, 'seed'):
        env.seed(seed)

    # sete attributes
    if isinstance(env, gym.Env):
        max_path_length = max_path_length or env.spec.max_episode_steps
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n if discrete else env.action_space.shape[0]
    else:
        if hasattr(env, 'state_size'):
            state_size = env.state_size
        else:
            raise Exception("Environment has attribute state_size or use gym.Env!")
        if hasattr(env, 'action_size'):
            action_size = env.action_size
        else:
            raise Exception("Environment has attribute action_size or use gym.Env!")
    
    net_args = {
        "n_layers": n_layers,
        "state_size": state_size,
        "action_size": action_size,
        "discrete": discrete,
        "hidden_size": hidden_size,
        "learing_rate": lr,
        "output_activation": nn.Sigmoid()
    }

    trajectory_args = {
        "max_path_length": max_path_length,
        "min_timesteps_per_batch": min_timesteps_per_batch
    }

    reward_args = {
        "gamma": gamma,
        "nn_baseline": nn_baseline,
        "normalize_advantage": normalize_advantages
    }

    agent = Agent(net_args, trajectory_args, reward_args)

    # create networks 
    agent.build_net()

    total_timesteps = 0
    for it in range(n_iters):
        print("=============Iteration {}==============".format(it))
        paths, timesteps_this_batch = agent.sample_trajectories(it, env)
        #TODO:
        # env = ChallengeSeqDecEnvironment(experimentCount=3005, userID="jingw2", \
        #     timeout=5, realworkercount=8)
        total_timesteps += timesteps_this_batch

        states = np.concatenate([path["state"] for path in paths])
        actions = np.concatenate([path["action"] for path in paths])
        rewards = [path["reward"] for path in paths]

        states_input = torch.Tensor(states).float()
        actions_input = torch.Tensor(actions).float()
        # q_n, adv = agent.estimate_return(states_input, rewards)
        # agent.train_op(states_input, actions_input, q_n, adv)
        agent.train_op()

        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]

        # best_idx = np.argmax(returns)
        # best_path = paths[best_idx]
        # best_policy = {}
        # for i in range(5):
        #     best_policy[str(i+1)] = best_path["action"][i].tolist()
        # data = {"best_policy": [best_policy], "best_reward": returns[best_idx]}
        # data = pd.DataFrame(data)
        # if os.path.exists("best_policy_pg.csv"):
        #     policy_df = pd.read_csv("best_policy_pg.csv")
        #     policy_df.loc[len(policy_df)] = [best_policy, returns[best_idx]]
        # else:
        #     policy_df = data
        # policy_df.to_csv("best_policy_pg.csv", index=False)

        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", it)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--no_time', '-nt', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--hidden_size', '-hs', type=int, default=64)
    parser.add_argument('--pg_step', '-ps', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # check gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)


    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iters=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                lr=args.learning_rate,
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                hidden_size=args.hidden_size,
                discrete=False,
                logdir=os.path.join(logdir, '%d' % seed)
                )
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # train_func()
        # if you comment in the line below, then the loop will block 
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
