import numpy as np
import arms
import matplotlib.pyplot as plt


def bandit_game(bandits, T, strategy='thompson', rho=.2):
    if strategy not in ['thompson', 'ucb1']:
        raise NameError('Unknown strategy')
    n = len(bandits)
    N = np.zeros((n, 1))
    R = np.zeros((n, 1))
    S = np.zeros((n, 1))
    F = np.zeros((n, 1))
    draws = np.zeros((T, 1))
    rewards = np.zeros((T, 1))
    arm = -1
    for t in range(T):

        if strategy is 'ucb1':
            if t < n:
                arm = t
            else:
                arm = np.argmax(R / N + rho * np.sqrt(.5 * np.log(t) / N))

        if strategy is 'thompson':
            arm = np.random.beta(S + 1, F + 1).argmax()

        reward = bandits[arm].sample()
        draws[t] = arm
        N[arm] += 1
        if np.random.random() < reward:
            S[arm] += 1
        else:
            F[arm] += 1
        R[arm] += reward
        rewards[t] = reward

    return rewards, draws


def avg_bandit_game(bandits, T, strategy='thompson', rho=.2, runs=20):
    return np.array([np.array(bandit_game(bandits, T, strategy=strategy, rho=rho))[:, :, 0] for i in range(runs)]).mean(
        axis=0)


# Build your own bandit problem

# this is an example, please change the parameters or arms!
arm1 = arms.ArmBernoulli(0.65, random_state=np.random.randint(1, 312414))
arm2 = arms.ArmBernoulli(0.5, random_state=np.random.randint(1, 312414))
arm3 = arms.ArmBernoulli(0.45, random_state=np.random.randint(1, 312414))
arm4 = arms.ArmBernoulli(0.60, random_state=np.random.randint(1, 312414))

MAB = [arm1, arm2, arm3, arm4]

arm21 = arms.ArmBernoulli(0.43, random_state=np.random.randint(1, 312414))
arm22 = arms.ArmBernoulli(0.56, random_state=np.random.randint(1, 312414))
arm23 = arms.ArmBernoulli(0.51, random_state=np.random.randint(1, 312414))
arm24 = arms.ArmBernoulli(0.55, random_state=np.random.randint(1, 312414))

MAB2 = [arm21, arm22, arm23, arm24]

for mab in [MAB, MAB2]:
    # bandit : set of arms

    nb_arms = len(mab)
    means = [el.mean for el in mab]

    # Display the means of your bandit (to find the best)
    print('means: {}'.format(means))
    mu_max = np.max(means)

    # Comparison of the regret on one run of the bandit algorithm
    # try to run this multiple times, you should observe different results

    T = 10000  # horizon

    rew1, draws1 = avg_bandit_game(mab, T, strategy='ucb1', runs=100)
    reg1 = mu_max * np.arange(1, T + 1) - np.cumsum(rew1)
    rew2, draws2 = avg_bandit_game(mab, T, strategy='thompson', runs=100)
    reg2 = mu_max * np.arange(1, T + 1) - np.cumsum(rew2)

    # reg3 = naive strategy
    best_arm = mab[np.argmax(means)]
    # reg3 = mu_max * np.arange(1, T + 1) - np.cumsum(
    #     np.array([[best_arm.sample() for i in range(T)] for r in range(10)]).mean(axis=0))

    # add oracle t -> C(p)log(t)
    kl = lambda x, y: x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))
    C = np.array([(mu_max - m) / kl(m, mu_max) for m in means if m != mu_max]).sum()
    oracle = C * np.log(np.arange(1, T + 1))

    plt.figure(1)
    x = np.arange(1, T + 1)
    plt.plot(x, reg1, label='UCB')
    plt.plot(x, reg2, label='Thompson')
    # plt.plot(x, reg3, label='Best arm')
    plt.plot(x, oracle, label='Oracle')
    plt.legend(['UCB', 'Thompson', 'Oracle'])
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Regret')
    # plt.title('First problem')

    plt.show()
# (Expected) regret curve for UCB and Thompson Sampling

npm_1 = arms.ArmBeta(0.7, 0.6)
npm_2 = arms.ArmBeta(0.5, 0.6)
npm_3 = arms.ArmExp(0.7)
npm_4 = arms.ArmExp(0.35)
NPM = [npm_1, npm_2, npm_3]

means = [el.mean for el in NPM]
mu_max = np.max(means)
rew4, draws4 = avg_bandit_game(NPM, T, strategy='ucb1', runs=100)
reg4 = mu_max * np.arange(1, T + 1) - np.cumsum(rew4)
rew5, draws5 = avg_bandit_game(NPM, T, strategy='thompson', runs=100)
reg5 = mu_max * np.arange(1, T + 1) - np.cumsum(rew5)
plt.figure(1)
x = np.arange(1, T + 1)
plt.plot(x, reg4, label='UCB')
plt.plot(x, reg5, label='Thomson')
plt.legend(['UCB', 'Thompson'])
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.show()
