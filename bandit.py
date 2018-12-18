import random
import math
import numpy as np

class Bandit(object):
    def __init__(self, k, mean=0, variance=1, nonstationary=False):
        self.k = k
        self.mean = mean
        self.variance = variance
        self.nonstationary = nonstationary
        self.arms = np.random.normal(mean, variance, size=k)

    def step(self, action):
        reward = np.random.normal(self.arms[action], 1)
        if self.nonstationary:
            self.add_noise()
        return reward

    def add_noise(self, mean=0, variance=.01):
        self.arms += np.random.normal(mean, variance, size=self.k)
