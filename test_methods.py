import gym
import utils
import lake_env as lake_env 
import numpy as np


def test_methods():		
	env = gym.make('Stochastic-4x4-FrozenLake-v0')   #  Create Stochastic Frozen Lake Environment

	tester = utils.Tester()  # Construct Testing Object

	gamma = 0.99
	init_policy = np.ones([env.nS,], dtype =np.int32)*2

	values,l = tester.evaluate_policy(env, gamma, init_policy)
	print(values,l)

	values,k = tester.value_iteration(env, gamma=0.99)
	print(values,k)

	policy,values,i,j = tester.policy_iteration(env, gamma=0.99)  # Run Value Iteration
	print(values, i,j)
	print(policy)

def main():
	test_methods()
	
if __name__ == '__main__':
	main()