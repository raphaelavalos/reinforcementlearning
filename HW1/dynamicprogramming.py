import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model1_R = np.zeros(((3, 3)))
model1_R[0, 2] = 5 / 100
model1_R[2, 1] = 1
model1_R[2, 2] = 9 / 10

model1_T = np.zeros(((3, 3, 3)))
model1_T[0, 0, 0] = .55
model1_T[0, 0, 1] = .45
model1_T[0, 1, 0] = .3
model1_T[0, 1, 1] = .7
model1_T[0, 2, 0] = 1
model1_T[1, 0, 0] = 1
model1_T[1, 1, 1] = .4
model1_T[1, 1, 2] = .6
model1_T[1, 2, 1] = 1
model1_T[2, 0, 1] = 1
model1_T[2, 1, 1] = .6
model1_T[2, 1, 2] = .4
model1_T[2, 2, 2] = 1


def value_iteration(transition_matrix, reward_matrix, epsilon=.01, gamma=.95, plot=False):
    '''
    Compute the epsilon optimal policy by the value iteration algorithm
    :param transition_matrix: transition_matrix[state,action,state2] = p(state2|state,action)
    :type transition_matrix: numpy.ndarray
    :param reward_matrix: reward_matrix[state,action] = reward(state,action)
    :type reward_matrix: numpy.ndarray
    :param epsilon: epsilon value
    :type epsilon: float
    :param gamma: discount factor
    :type gamma: float
    :param plot: if True plot the improvements over the iteration
    :type plot: bool
    :return: policy[state]=action
    :rtype: numpy.array

    '''
    n = transition_matrix.shape[0]
    value = np.random.random((n, 1))
    update = epsilon + 1
    values = list()
    while update >= epsilon:
        tmp = np.max(reward_matrix + gamma * (transition_matrix @ value)[:, :, 0], axis=1).reshape((n, 1))
        update = np.linalg.norm(value - tmp, ord=np.inf)
        value = tmp
        values.append(value)
    policy = np.argmax(reward_matrix + gamma * (transition_matrix @ value)[:, :, 0], axis=1)
    if plot:
        plt.scatter(x=list(range(1, len(values))), y=np.linalg.norm(values[1:] - value, ord=np.inf, axis=1))
        plt.show()
    return policy, value


def policy_iteration(transition_matrix, reward_matrix, gamma=.95):
    '''
    Compute the exact policy by the policy iteration algorithm
    :param transition_matrix: transition_matrix[state,action,state2] = p(state2|state,action)
    :type transition_matrix: numpy.ndarray
    :param reward_matrix: reward_matrix[state,action] = reward(state,action)
    :type reward_matrix: numpy.ndarray
    :param gamma: discount factor
    :type gamma: float
    :return: policy[state]=action
    :rtype: numpy.array
    '''
    n = transition_matrix.shape[0]
    a = transition_matrix.shape[1]
    policy = np.full((3,), 0)
    value = np.random.randint(n, size=(n, 1))
    states = np.arange(n)
    while True:
        tmp = np.linalg.inv(np.identity(n) - gamma * transition_matrix[states, policy[states]]) @ reward_matrix[
            states, policy]
        tmp = tmp.reshape((n, 1))
        if (tmp == value).all():
            break
        value = tmp
        policy = np.argmax(reward_matrix + gamma * (transition_matrix @ value)[:, :, 0], axis=1)
    policy = np.argmax(reward_matrix + gamma * (transition_matrix @ value)[:, :, 0], axis=1)
    return policy
