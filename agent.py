import math
import random
import numpy as np

class Agent(object):
    def __init__(self, k, alpha):
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.k = k
        self.alpha = alpha

    def choose_action(self):
        raise NotImplementedError

    def update(self, action, reward):
        raise NotImplementedError

    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)


class GreedyAgent(Agent):
    def __init__(self, k, alpha=.01, optimism=0):
        super().__init__(k, alpha)
        self.Q += np.full(k, optimism)

    def choose_action(self):
        return np.argmax(self.Q)

    def update(self, action, reward):
        self.Q[action] += self.alpha*(reward-self.Q[action])
        self.N[action] += 1


class UcbAgent(Agent):
    def __init__(self, k, alpha, c):
        super().__init__(k, alpha)
        self.t = 1
        self.c = c

    def choose_action(self):
        for i in range(len(self.Q)):
            if self.N[i] == 0:
                self.t += 1
                return i
        ucb = self.Q + self.c * (math.log(self.t)/self.N)
        self.t += 1
        return np.argmax(ucb)

    def update(self, action, reward):
        self.Q[action] += self.alpha*(reward-self.Q[action])
        self.N[action] += 1


class EpsilonGreedyAgent(Agent):
    def __init__(self, k, alpha, epsilon):
        super().__init__(k, alpha)
        self.epsilon = epsilon

    def choose_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.Q)-1)
        else:
            return np.argmax(self.Q)
    
    def update(self, action, reward):
        self.Q[action] += self.alpha*(reward-self.Q[action])
        self.N[action] += 1


class GradientAgent(Agent):
    def __init__(self, k, alpha):
        super().__init__(k, alpha)
        self.H = np.zeros(k)
        self.P = np.full(k, 1/k)
        self.baseline = 0
        self.n = 0

    def choose_action(self):
        action = np.random.choice(len(self.P), p=self.P)
        return action

    def update(self, action, reward):
        self.n += 1
        self.baseline += (1/self.n)*(reward-self.baseline)
        a = self.alpha
        b = self.baseline
        update = [a*(reward-b)*(1-self.P[i]) if i == action else -a*(reward-b)*(self.P[i]) for i in range(len(self.H))]
        self.H += update
        self.update_probabilities()
    
    def update_probabilities(self):
        exp_h = np.exp(self.H-np.max(self.H))
        self.P = exp_h / np.sum(exp_h, axis=0)