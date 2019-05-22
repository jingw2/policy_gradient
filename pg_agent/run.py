#!/usr/bin/python 3.6
# -*-coding:utf-8-*-

import sys
sys.path.append('../')
import time
import inspect
import os
from multiprocessing import Process
import numpy as np
import pandas as pd

# self-defineed
import utils.logz as logz
import ddpg
import sac
from agent import Agent

# torch
import torch
import torch.nn as nn

from netsapi.challenge import *

# gym
import gym

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
    logdir,
    method,
    method_args):

    start = time.time()

    # env
    env = gym.make(env_name)
    #TODO:
    # env = ChallengeSeqDecEnvironment(experimentCount=3005, userID="jingw2", \
    #     timeout=5, realworkercount=4)
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
        "output_activation": None
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

    if method == "sac":
        agent = sac.SAC(net_args, trajectory_args, reward_args, method_args)
    elif method == "ddpg":
        agent = ddpg.DDPG(net_args, trajectory_args, reward_args, method_args)
    elif method == "vanilla":
        agent = Agent(net_args, trajectory_args, reward_args)

    # create networks 
    agent.build_net()

    total_timesteps = 0
    for it in range(n_iters):
        print("=============Iteration {}==============".format(it))
        paths, timesteps_this_batch = agent.sample_trajectories(it, env)
        #TODO:
        # env = ChallengeSeqDecEnvironment(experimentCount=3005, userID="jingw2", \
        #     timeout=5, realworkercount=4)
        total_timesteps += timesteps_this_batch

        states = np.concatenate([path["state"] for path in paths])
        actions = np.concatenate([path["action"] for path in paths])
        rewards = [path["reward"] for path in paths]
        # next_states = np.concatenate([path["next_state"] for path in paths])

        states_input = torch.Tensor(states).float()
        actions_input = torch.Tensor(actions).float()
        if method == "vanilla":
            q_n, adv = agent.estimate_return(states_input, rewards)
            agent.train_op(states_input, actions_input, q_n, adv)
        else:
            agent.train_op()

        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]

        best_idx = np.argmax(returns)
        best_path = paths[best_idx]
        best_policy = {}
        for i in range(5):
            best_policy[str(i+1)] = best_path["action"][i].tolist()
        data = {"best_policy": [best_policy], "best_reward": returns[best_idx]}
        data = pd.DataFrame(data)
        if os.path.exists("best_policy_pg.csv"):
            policy_df = pd.read_csv("best_policy_pg.csv")
            policy_df.loc[len(policy_df)] = [best_policy, returns[best_idx]]
        else:
            policy_df = data
        policy_df.to_csv("best_policy_pg.csv", index=False)

        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", it)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        # logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
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
    parser.add_argument('--method', '-m', type=str, default="vpg")
    # sac argument
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--duel_q_net', '-dq', action='store_true')
    parser.add_argument('--policy_type', '-pt', type=str, default="gaussian")
    parser.add_argument('--action_bound_fn', '-abf', type=str, default="tanh")

    # ddpg argument
    parser.add_argument('--ounoise', action='store_true')
    parser.add_argument('--decay', action='store_true')

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

    if args.method == "sac":
        method_args = {
            "tau": args.tau,
            "duel_q_net": args.duel_q_net,
            "policy_type": args.policy_type,
            "action_bound_fn": args.action_bound_fn
        }
    elif args.method == "ddpg":
        method_args = {
            "tau": args.tau,
            "ounoise": args.ounoise,
            "decay": args.decay
        }

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
                logdir=os.path.join(logdir, '%d' % seed),
                method=args.method,
                method_args=method_args
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
