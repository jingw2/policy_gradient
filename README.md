# policy gradient

This section is about policy gradient method, including simple policy gradient method and trust region policy optimization.
Based on cart-v0 environment from openAI gym module, two methods are implemented using pytorch. 


### Policy gradient theory
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

one fixed policy + random part. We can use different functions to approximate the fixed part. Random part can be normal distribution. 

One training curve is as follows

<img src="https://github.com/jingw2/policy_gradient/blob/master/pg.png" width="300">

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
![linesearch](https://github.com/jingw2/policy_gradient/blob/master/linesearch.gif)

where ei is expected improvement 

One training curve in cart-v0 as follows:

<img src="https://github.com/jingw2/policy_gradient/blob/master/trpo.png" width="300">


### Reference link:
* https://github.com/gxnk/reinforcement-learning-code/blob/master/
* https://github.com/mjacar/pytorch-trpo
* [Trust Region Policy Optimization](http://proceedings.mlr.press/v37/schulman15.pdf)
* [Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-217.pdf)
