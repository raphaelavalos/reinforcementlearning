import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def monte_carlo_evaluation(env, policy, max_iter=1000, t_max=100, gamma=0.95):
    v = np.zeros((max_iter + 1, env.n_states))
    n = np.zeros(env.n_states)
    for i in range(1, max_iter + 1):
        s0 = env.reset()
        n[s0] += 1
        s = s0
        cummulated_reward = 0
        t = 0
        term = False
        while not term or t < t_max:
            t += 1
            a = policy[s]
            s, r, term = env.step(s, a)
            cummulated_reward += r * gamma ** (t - 1)
        v[i] = v[i - 1]
        v[i][s0] = ((n[s0] - 1) * v[i][s0] + cummulated_reward) / n[s0]
    return v


def estimate_starting_state(env, max_iter=200):
    mu = np.zeros(env.n_states)
    for i in range(max_iter):
        mu[env.reset()] += 1
    mu /= max_iter
    return mu


def build_j(mu, v):
    return (mu * v).sum(axis=1)


def plot_diff_j(mu, v, v_pi):
    j_n = build_j(mu, v)
    j_pi = build_j(mu, v_pi)
    diff = j_n - j_pi
    plt.plot(diff)
    plt.show()


def policy_optimization(env, epsilon=0.7, max_iter=1000, t_max=100, gamma=0.95):
    N = np.zeros((env.n_states, len(env.action_names)))
    Q = np.zeros((max_iter * t_max, env.n_states, len(env.action_names)))
    R = np.zeros(max_iter * t_max)
    R_cummulated = np.zeros(max_iter)
    t = 0
    for i in range(max_iter - 1):
        t_i = 0
        s0 = env.reset()
        state = s0
        term = False
        while t_i < t_max and not term:
            Q[t + 1] = np.copy(Q[t])
            possible_actions = env.state_actions[state]
            if len(possible_actions) == 1:
                action = possible_actions[0]
            else:
                if np.random.rand() < epsilon:
                    action = possible_actions[Q[i, state, possible_actions].argmax()]
                else:
                    action = np.random.choice(possible_actions)
            next_state, reward, term = env.step(state, action)
            N[state, action] += 1
            Q[t + 1, state, action] = (1 - 1 / N[state, action]) * Q[t + 1, state, action] + 1 / N[state, action] * (
                    reward + gamma * Q[t, next_state].max())
            state = next_state
            R[t] = reward
            R_cummulated[i] += reward
            t += 1
            t_i += 1
    Q, R = Q[:t + 1], R[:t + 1]
    policy = Q.argmax(axis=2)
    return policy, Q, R, R_cummulated


def compare_value_function(values_n, value, title=""):
    plt.plot(np.abs(values_n - value).max(axis=1))
    plt.title(title)
    plt.show()
