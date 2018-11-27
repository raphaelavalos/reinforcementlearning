import numpy as np


class BaseAlgorithm(object):

    def __init__(self, model, T=1000, lamb=1):
        self.model = model
        self.d = model.n_features
        self.arms = model.n_actions
        self.T = T
        self.lamb = 1
        self.theta = np.zeros((self.d, 1))
        self.A = self.lamb * np.identity(self.d)
        self.b = np.zeros((self.d, 1))

    def pull(self):
        pass

    def init(self):
        self.theta = np.zeros((self.d, 1))
        self.A = self.lamb * np.identity(self.d)
        self.b = np.zeros((self.d, 1))

    def run(self):
        self.init()
        return np.array([np.array(self.pull()) for i in range(self.T)])

    def average(self, n):
        return np.array([self.run() for i in range(n)]).mean(axis=0)


class Epsilon(BaseAlgorithm):

    def __init__(self, model, T=1000, epsilon=0.5, lamb=1):
        super(Epsilon, self).__init__(model, T, lamb)
        self.epsilon = epsilon
        self.lamb = 1

    def pull(self):
        inv_A = np.linalg.inv(self.A)
        self.theta = inv_A @ self.b
        if np.random.rand() < self.epsilon:
            arm = np.argmax(self.model.features @ self.theta)
        else:
            arm = np.random.randint(self.model.n_actions)
        reward = self.model.reward(arm)
        self.A += self.model.features[arm].reshape(-1, 1) @ self.model.features[arm].reshape(1, -1)
        self.b += reward * self.model.features[arm].reshape(-1, 1)
        return arm, self.model.best_arm_reward() - reward, np.linalg.norm(
            self.theta.reshape(-1) - self.model.real_theta.reshape(-1), ord=2)


class Random(Epsilon):

    def __init__(self, model, T=1000, lamb=1):
        super(Random, self).__init__(model, T, epsilon=-1, lamb=lamb)


class LinearUCB(BaseAlgorithm):

    def __init__(self, model, T=1000, lamb=1, alpha=100):
        super(LinearUCB, self).__init__(model, T, lamb)
        self.alpha = alpha
        self.a = alpha

    def init(self):
        super(LinearUCB, self).init()
        self.a = self.alpha

    def pull(self):
        inv_A = np.linalg.inv(self.A)
        self.theta = inv_A @ self.b
        beta = self.a * np.sqrt(np.diag(self.model.features @ inv_A @ self.model.features.T)).reshape(-1, 1)
        arm = np.argmax(self.model.features @ self.theta + beta)
        reward = self.model.reward(arm)
        self.A += self.model.features[arm].reshape(-1, 1) @ self.model.features[arm].reshape(1, -1)
        self.b += reward * self.model.features[arm].reshape(-1, 1)
        #self.a = max(1, self.a - 1)
        return arm, self.model.best_arm_reward() - reward[0], np.linalg.norm(
            self.theta.reshape(-1) - self.model.real_theta.reshape(-1), ord=2)
