import matplotlib.pyplot as plt
import seaborn as sns

def monte_carlo_evaluation(env, policy, max_iter=1000, t_max = 100, gamma=0.95):
    v = np.zeros((max_iter+1,env.n_states))
    n = np.zeros(env.n_states)
    for i in range(1,max_iter+1):
        s0 = env.reset()
        n[s0] += 1
        s = s0
        cummulated_reward = 0
        t = 0
        term = False
        while not term or t<t_max:
            t +=1
            a = policy[s]
            s, r, term = env.step(s,a)
            cummulated_reward += r * gamma**(t-1)
        v[i] = v[i-1]
        v[i][s0] = ((n[s0] - 1)*v[i][s0] + cummulated_reward)/n[s0]
    return v

def estimate_starting_state(env, max_iter=200):
    mu = np.zeros(env.n_states)
    for i in range(max_iter):
        mu[env.reset()] += 1
    mu /= max_iter
    return mu

def build_j(mu,v):
    return (mu * v).sum(axis=1)

def plot_diff_j(mu, v, v_pi):
    j_n = build_j(mu, v)
    j_pi = build_j(mu,v_pi)
    diff = j_n - j_pi
    plt.plot(diff)
    plt.show()

def policy_optimization(env, policy, max_iter=1000):
    pass