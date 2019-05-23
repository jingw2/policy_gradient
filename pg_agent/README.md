## Policy Gradient Agents Combination

This folder is to create policy gradient agents combined with different policy gradient algorithms, 
including:
* Vanilla Policy Gradient (VPG)
* Trust Region Policy Optimization (TRPO)
* Proximal Policy Optimization (PPO)
* Deep Deterministic Policy Gradient (DDPG)
* Soft Actor-Critic (SAC)
* TD3

to support different methods running in Gym environments or others. 

### Run examples
```shell
python run.py env_name MountainCarContinuous-v0 --discount 0.99 -n 100 
  -b 20000 -ep 1000 -lr 1e-2 -m vpg -e 1 -hs 256 -dna
```

### Arguments
We have different arguments to support environment arguments, network arguments, trajectory arguments, reward arguments,
and method arguments.

#### Environment arguments
| name          | explanation   | default  |
| ------------- |:-------------:| -----:|
| env_name      | gym environment name | None |
| --exp_name      | output folder name |   vpg |
| --render | show figure      |  apply --render to use |

#### experiment arguments
| name          | explanation   | default  |
| ------------- |:-------------:| -----:|
| --n_iter (-n)      | number of iterations | 100 |
| --batch_size (-b)      | batch_size      |  1000 |
| --ep_len (-ep)      | one episode length      |  max episode length <br> of enviroment|
| --learning_rate (-lr) |learning rate  | 5e-3 |
| --n_experiments (-e) | number of experiments | 1 |
| --seed | random seed | 1 |
| --gpu | control gpu use | 1 |
| --method (-m) | method to run | vpg |

#### VPG Arguments
| name          | explanation   | default  |
| ------------- |:-------------:| -----:|
| --dont_normalize_advantages (-dna) | as shows      | apply -dna to use|
| --nn_baseline (-bl) | use baseline to <br> loss function | apply -bl to use|

PG Agents support the asychronous actor critic algorithm because multiprocessing there. To use, just apply -bl. 


#### DDPG Arguments
| name          | explanation   | default  |
| ------------- |:-------------:| -----:|
| --tau          | soft update coefficient   | 0.005  |
| --ounoise      | use ounoise to sample actions   | apply --ounoise to use  |
| --decay        | use decay to control ounoise   | apply --decay to use  |

#### SAC Arguments
| name          | explanation   | default  |
| ------------- |:-------------:| -----:|
| --tau          | soft update coefficient   | 0.005  |
| --duel_q_net (-dq)     | duel Q network to use   | apply -dq to use  |
| --policy_type (-pt)        | gaussian/deterministic   | gaussian  |
| --action_bound_fn (-abf)    | enforce action bound   | tanh  |

### TO DO: 
* [ ] Add tensorboard
* [ ] Add model save and load, args
* [ ] Add requirements.txt
* [ ] Show some results
* [ ] Add PPO
* [ ] Add TRPO
* [ ] Add TD3
* [ ] Add parameter noise

### Reference
* [cs294-112](http://rail.eecs.berkeley.edu/deeprlcourse/)
* [Deep-reinforcement-learning-with-pytorch](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch)
* [Pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic)
* [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)
* [Soft Actor Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)
* [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf)
