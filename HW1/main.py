from gridworld import GridWorld1
import gridrender as gui
import numpy as np
import timeit
import dynamicprogramming
import policy_tools
import matplotlib.pyplot as plt
import seaborn as sns


# Wrapper for timeit
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


# VI
policy, value = dynamicprogramming.value_iteration(dynamicprogramming.model1_T, dynamicprogramming.model1_R, plot=True)
print('Value iteration policy : ', policy)
print('Value iteration value : ', value)

# PI
policy = dynamicprogramming.policy_iteration(dynamicprogramming.model1_T, dynamicprogramming.model1_R)
print('Policy iteration policy : ', policy)

l = wrapper(dynamicprogramming.value_iteration, dynamicprogramming.model1_T, dynamicprogramming.model1_R, plot=False)
t = timeit.timeit(l, number=100) / 100
print('VI mean time: ', t)

ll = wrapper(dynamicprogramming.policy_iteration, dynamicprogramming.model1_T, dynamicprogramming.model1_R)
t = timeit.timeit(l, number=100) / 100
print('PI mean time : ', t)

# Part 2
env = GridWorld1
v_q4 = np.array(
    [[0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
      -0.93358351, -0.99447514]])
pol = [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3]

# Monte carlo
v = policy_tools.monte_carlo_evaluation(env, pol)
mu = policy_tools.estimate_starting_state(env)
policy_tools.plot_diff_j(mu, v, v_q4)

# Q learning


v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
         0.87691855, 0.82847001]
for epsilon in [.99, .95, .90, 85, 80]:
    for j in range(3):
        policy, Q, R, R_cummulated = policy_tools.policy_optimization(env, epsilon=epsilon, t_max=200, max_iter=5000)
        policy_tools.compare_value_function(Q.max(axis=2), v_opt, title="Epsilon : %.2f, Run: %i" % (epsilon, j))
        # plt.scatter(list(range(len(R_cummulated))), R_cummulated)
        # plt.show()
