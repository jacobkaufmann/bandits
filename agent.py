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
        old = self.Q[action]
        self.Q[action] = old+self.alpha*(reward-old)
        self.N[action] += 1

    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)


class GreedyAgent(Agent):
    def __init__(self, k, alpha=.01, optimism=0):
        super().__init__(k, alpha)
        self.Q += np.full(k, optimism)

    def choose_action(self):
        return np.argmax(self.Q)


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


class EpsilonGreedyAgent(Agent):
    def __init__(self, k, alpha, epsilon):
        super().__init__(k, alpha)
        self.epsilon = epsilon

    def choose_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.Q)-1)
        else:
            return np.argmax(self.Q)
