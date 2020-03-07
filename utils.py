import numpy as np
import gym

class Tester(object):

    def __init__(self):
        values = np.zeros(16)

    def evaluate_policy(self, env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
        """Evaluate the value of a policy.

        See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        policy: np.array
          The policy to evaluate. Maps states to actions.
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.
        
        Returns
        -------
        np.ndarray, iteration
          The value function and the number of iterations it took to converge.
        """
        # TODO: Your Code Goes Here
        
        iterations = max_iterations
        numStates = env.nS
        numActions = env.nA
        

        # values = self.values
        values = np.zeros(16)
        old_values = np.zeros(16)
        delta = 1000

        while(iterations>0 and delta>tol):
          iterations -= 1  
          
          for state in range(numStates):
            action = policy[state]
            p,newstate,reward,isterminal = np.array(env.P[state][action]).T
            newstate = np.array(newstate,dtype=np.int32)
            isterminal = np.array(isterminal,dtype=np.bool)
            
            values[state] = np.dot(p,(reward + gamma*values[newstate]))

          delta = np.amax(np.absolute(values-old_values))
          # print(values,delta)
          old_values = np.copy(values)  

        num_evaluations = max_iterations-iterations  
        return values, num_evaluations

    def policy_iteration(self, env, gamma, max_iterations=int(1e3), tol=1e-3):
        """Runs policy iteration.

        See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        You should use the improve_policy and evaluate_policy methods to
        implement this method.

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.

        Returns
        -------
        (np.ndarray, np.ndarray, int, int)
           Returns optimal policy, value function, number of policy
           improvement iterations, and number of value iterations.
        """
        # TODO:  Your code goes here.
        iterations = max_iterations
        numStates = env.nS
        numActions = env.nA

        old_policy = np.ones([env.nS,], dtype =np.int32)*2
        policy = np.ones([env.nS,], dtype =np.int32)*2
        delta = 1000
        
        while(iterations>0 and delta!=0):
          iterations -= 1
          
          values,num_evaluations = self.evaluate_policy(env, gamma, policy, max_iterations, tol)
          # print(values,num_evaluations)  
          
          qvalues = np.zeros(numActions)

          for state in range(numStates):
            for action in range(numActions):
              p,newstate,reward,isterminal = np.array(env.P[state][action]).T
              newstate = np.array(newstate,dtype=np.int32)
              isterminal = np.array(isterminal,dtype=np.bool)
              
              qvalues[action] = np.dot(p,(reward + gamma*values[newstate]))
            
            policy[state] = np.argmax(qvalues) 
          
          delta = np.sum(policy-old_policy)
          # print(delta)
          old_policy = np.copy(policy)  
        
        num_policy_iterations = max_iterations-iterations

        return policy, values, num_policy_iterations, num_evaluations

    def value_iteration(self, env, gamma, max_iterations=int(1e3), tol=1e-3):
        """Runs value iteration for a given gamma and environment.

        See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.

        Returns
        -------
        np.ndarray, iteration
          The value function and the number of iterations it took to converge.
        """
        # TODO: Your Code goes here.
        iterations = max_iterations
        numStates = env.nS
        numActions = env.nA
        
        values = np.zeros(numStates)
        old_values = np.zeros(numStates)
        delta = 1000

        while(iterations>0 and delta>tol):
          iterations -= 1  
          # print(iterations)
          qvalues = np.zeros(numActions)

          for state in range(numStates):
            for action in range(numActions):
              p,newstate,reward,isterminal = np.array(env.P[state][action]).T
              newstate = np.array(newstate,dtype=np.int32)
              isterminal = np.array(isterminal,dtype=np.bool)
              
              qvalues[action] = np.dot(p,(reward + gamma*values[newstate]))

            values[state] = np.amax(qvalues)

          delta = np.amax(values-old_values)
          # print(delta)
          old_values = np.copy(values)  

        value_iterations = max_iterations-iterations  
        return values, value_iterations
