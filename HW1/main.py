from gridworld import GridWorld1
import gridrender as gui
import numpy as np
import time
import dynamicprogramming
import policy_tools
import matplotlib.pyplot as plt
import seaborn as sns

policy = dynamicprogramming.value_iteration(dynamicprogramming.model1_T, dynamicprogramming.model1_R, plot=True)
print(policy)
policy = dynamicprogramming.policy_iteration(dynamicprogramming.model1_T, dynamicprogramming.model1_R)
print(policy)

env = GridWorld1
v_q4 = np.array([[0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
        -0.93358351, -0.99447514]])

pol = [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3]


policy, Q, R, R_cummulated = policy_tools.policy_optimization(env)

v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
         0.87691855, 0.82847001]
policy_tools.compare_value_function(Q.max(axis=2), v_opt)

