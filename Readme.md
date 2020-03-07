Basic RL examples 
Contains python code examples for policy_evaluation, policy_iteration, value_iteration (tested on Stochastic-4x4-FrozenLake-v0)

lake_env.py - Defines some frozen lake maps

utils.py - You need to define environment (Eg- lake_env.py) and then call individual methods which return the final values for each state, the optimal policy and also the number of iterations it takes for convergence. 

test_methods.py - This is the code for testing all the three 3 methods individually. 

The optimal policy to solve the above environment:

[0 3 3 3 0 0 0 0 3 1 0 0 0 2 1 0]

where, {0: 'LEFT', 1: 'RIGHT', 2: 'DOWN', 3: 'UP'}
