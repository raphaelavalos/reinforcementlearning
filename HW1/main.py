import dynamicprogramming

policy = dynamicprogramming.value_iteration(dynamicprogramming.model1_T, dynamicprogramming.model1_R, plot=True)
print(policy)
policy = dynamicprogramming.policy_iteration(dynamicprogramming.model1_T, dynamicprogramming.model1_R)
print(policy)