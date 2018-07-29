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
