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

![equation](http://www.sciweavers.org/tex2img.php?eq=J%28%5Ctheta%29%20%3D%20%5Cmathbb%7BE%7D_%7Bs%20%5Csim%20%5Crho%5E%2A%2C%20a%20%5Csim%20%5Cpi_s%7D%20%5B%5Clog%20%5Cpi_%7B%5Ctheta%7D%20%28a%20%7C%20s%29%20Q%5E%2A%28s%2C%20a%29%5D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

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

![equation](http://www.sciweavers.org/tex2img.php?eq=r_t%28%5Ctheta%29%20%3D%20%5Cfrac%7B%5Cpi_%7B%5Ctheta%7D%28a_t%20%7C%20s_t%29%7D%7B%5Cpi_%7B%5Ctheta_%5Cold%7D%28a_t%20%7C%20s_t%29%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

In order to modify the objective to penalize changes to the policy that move the ratio away from 1, clipped surrogate objective, fixed KL and adaptive KL penalty coefficient methods have been came up in the paper. 

* Clipped surrogate objective

![equation](http://www.sciweavers.org/tex2img.php?eq=J%28%5Ctheta%29%20%3D%20%5Cmathbb%7BE%7D_t%20%5Cleft%5B%5Cmin%28r_t%28%5Ctheta%29A_t%2C%20clip%28r_t%28%5Ctheta%29%2C%201%20-%20%5Cepsilon%2C%201%20%2B%20%5Cepsilon%29A_t%20%5Cright%5D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

where ![equation](http://www.sciweavers.org/tex2img.php?eq=A_t%20%3D%20r_t%20%2B%20%5Cgamma%20r_%7Bt%2B1%7D%20%2B%20%5Ccdots%20%2B%20%5Cgamma%5E%7BT-t%2B1%7D%20r_%7BT-1%7D%20%2B%20%5Cgamma%5E%7BT-t%7D%20V%28s_%7BT%7D%29%20-%20V%28s_t%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

* Fixed kl and adaptive kl penalty coefficient

![equation](http://www.sciweavers.org/tex2img.php?eq=J%28%5Ctheta%29%20%3D%20%5Cmathbb%7BE%7D_t%20%5Cleft%5B%20r_t%28%5Ctheta%29%20A_t%20-%20%5Cbeta%20KL%5B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%28%5Ccdot%20%7C%20s_t%29%2C%20%5Cpi_%7B%5Ctheta%7D%28%5Ccdot%20%7C%20s_t%29%20%5Cright%5D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

where ![equation](http://www.sciweavers.org/tex2img.php?eq=KL%5B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%28%5Ccdot%20%7C%20s_t%29%2C%20%5Cpi_%7B%5Ctheta%7D%28%5Ccdot%20%7C%20s_t%29%5D%20%3D%20P%28%5Ctheta_%7Bold%7D%29%20%5Clog%20%5Cfrac%7BP%28%5Ctheta%29%7D%7BP%28%5Ctheta_%7Bold%7D%29%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)


### Reference link:
* https://github.com/gxnk/reinforcement-learning-code/blob/master/
* https://github.com/mjacar/pytorch-trpo
* [Trust Region Policy Optimization](http://proceedings.mlr.press/v37/schulman15.pdf)
* [Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-217.pdf)
* [PPO](https://arxiv.org/abs/1707.06347)
