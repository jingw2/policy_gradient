# Policy Gradient Algorithm

This section is about policy gradient method, including simple policy gradient method and trust region policy optimization.
Based on cart-v0 environment from openAI gym module, two methods are implemented using pytorch. 


### Vanilla Policy gradient theory
Policy gradient is directly to approximate the policy function, mapping the state to the action. It has the following advantages
compared to value function approximation:

* Parameterization is simpler, easy to converge
* feasible to solve when action space is huge or infinite
* random policy can ensemble exploration

of course, it has some shortcomings:

* local optimal
* big variable to evaluate single policy

The objective to optimize the expectation of rewards. By importance sampling, we want to calculate the policy gradient. Normally,
random policy can be expressed as 

![equation](https://latex.codecogs.com/gif.latex?\pi_{\theta}&space;=&space;\mu_{\theta}&space;&plus;&space;\epsilon)

one fixed policy + random part. We can use different functions to approximate the fixed part. Random part can be normal distribution. The loss function is as follows

![equation](https://latex.codecogs.com/gif.latex?J(\theta)&space;=&space;\mathbb{E}_{s&space;\sim&space;\rho*,&space;a&space;\sim&space;\pi_s}&space;[\log&space;\pi_{\theta}(a,&space;s)&space;Q(s,&space;a)])

One training curve is as follows

<img src="https://github.com/jingw2/policy_gradient/blob/master/pictures/vanilla_policy_gradient.png" width="300">

### Trust Region Policy Optimization (TRPO)
The theory of trpo can be seen in paper. Here, I want to show steps and explain the update method

* sample actions and trajectories
* calculate mean KL divergence and fisher vector product
* construct surrogate loss and line search method by conjugate gradient algorithm

Update steps:

* calculate advantage, surrogate loss, and policy gradient. 
* if no gradient, return
* update theta, try to update value function, and policy network

line search:

![linesearch](https://github.com/jingw2/policy_gradient/blob/master/pictures/linesearch.gif)

where ei is expected improvement 

One training curve in cart-v0 as follows:

<img src="https://github.com/jingw2/policy_gradient/blob/master/pictures/trpo.png" width="300">

From my experience in TRPO, it is sensitive to the quality of data. And based on [PPO](https://arxiv.org/abs/1707.06347) and my experiments, TRPO might perform very differently in multiple trials because it uses a hard constraint on KL divergence. Thus, in order to improve TRPO, PPO came in. 

### Proximal Policy Optimization

It is simpler and more stable than TRPO. Firstly, define the ratio

![equation](https://latex.codecogs.com/gif.latex?r_t(\theta)&space;=&space;\frac{\pi_{\theta}(a_t&space;|&space;s_t)}{\pi_{\theta_{old}(a_t&space;|&space;s_t)}})

In order to modify the objective to penalize changes to the policy that move the ratio away from 1, clipped surrogate objective, fixed KL and adaptive KL penalty coefficient methods have been came up in the paper. 

* Clipped surrogate objective

![equation](https://latex.codecogs.com/gif.latex?J(\theta)&space;=&space;\mathbb{E}_t&space;\left[&space;\min&space;(r_t(\theta))A_t,&space;clip(r_t(\theta),&space;1&space;-&space;\epsilon,&space;1&space;&plus;&space;\epsilon)A_t&space;\right&space;])

where ![equation](https://latex.codecogs.com/gif.latex?A_t&space;=&space;r_t&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\cdots&space;&plus;&space;\gamma^{T-t&plus;1}&space;r_{T-1}&space;&plus;&space;\gamma^{T-t}V(s_T)&space;-&space;V(s_t))

* Fixed KL and adaptive KL penalty coefficient

![equation](https://latex.codecogs.com/gif.latex?\begin{align*}&space;J(\theta)&space;&=&space;\mathbb{E}_t&space;\left[r_t(\theta)&space;A_t&space;-&space;\beta&space;KL[\pi_{\theta_{old}}(\cdot&space;|s_t),&space;\pi_{\theta}(\cdot&space;|s_t)]&space;\right&space;]&space;\\&space;KL[\pi_{\theta_{old}}(\cdot&space;|s_t),&space;\pi_{\theta}(\cdot&space;|s_t)]&space;&=&space;P(\theta_{old})&space;\log&space;\frac{P(\theta)}{P(\theta_{old})}&space;\\&space;d&space;&=\mathbb{E}_t&space;[KL[\pi_{\theta_{old}}(\cdot&space;|s_t),&space;\pi_{\theta}(\cdot&space;|s_t)]]&space;\\&space;\end{align*})

With respect to adaptive KL, we have d shown above, and then

![equation](https://latex.codecogs.com/gif.latex?\beta&space;=&space;\begin{cases}&space;\beta&space;/&space;2&space;\hspace{0.5cm}&space;\text{if}&space;\hspace{0.5cm}&space;d&space;<&space;d_{target}&space;/&space;1.5&space;\\&space;2\beta&space;\hspace{0.5cm}&space;\text{if}&space;\hspace{0.5cm}&space;d&space;>&space;d_{target}&space;\times&space;1.5&space;\end{cases})

The following graphs show the learning rewards, 

<p align = "center">
<img src="https://github.com/jingw2/policy_gradient/blob/master/pictures/ppo_clipped_surrogate.png" width="300">
<img src="https://github.com/jingw2/policy_gradient/blob/master/pictures/ppo_fixed_kl.png" width="300">
<img src="https://github.com/jingw2/policy_gradient/blob/master/pictures/ppo_adaptive_kl.png" width="300">
  </p>
  
 ### Actor-Critic and Deep Deterministic Policy Gradient 
 
 This is especially for continuous controls using deep reinforcement learning. The main goal is to find the deterministic policy from exploratory behavior policy. From my experience, it does not converge faster than PPO, and sometimes it is sensitive to the quality of data. 

Learning curve:
<p align = "center">
<img src="https://github.com/jingw2/policy_gradient/blob/master/pictures/ddpg.png" width="300">
  </p>
  
In order to better train the model, someone came up with the data filter to filter dirty data and get better model. 

### Reference link:
* https://github.com/gxnk/reinforcement-learning-code/blob/master/
* https://github.com/mjacar/pytorch-trpo
* [Trust Region Policy Optimization](http://proceedings.mlr.press/v37/schulman15.pdf)
* [Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-217.pdf)
* [PPO](https://arxiv.org/abs/1707.06347)
* [Deterministic Policy Gradient Algorithm](http://proceedings.mlr.press/v32/silver14.pdf)
* [DDPG](https://arxiv.org/abs/1509.02971)
