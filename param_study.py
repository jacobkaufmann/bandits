import numpy as np
import math 

from bandit import Bandit
import agent

def main():
    k = 10
    alpha = 0.1
    steps = 1000

    mean = 0
    variance = 1
    nonstationary = False

    param_vals = [2**i for i in range(-7, 3)]
    for v in param_vals:
        bandit = Bandit(k, mean, variance, nonstationary)
        greedy = agent.GreedyAgent(k, alpha=alpha, optimism=v)
        e_greedy = agent.EpsilonGreedyAgent(k, alpha=alpha, epsilon=v)
        ucb = agent.UcbAgent(k, alpha=alpha, c=v)
        names = ["Greedy", "Epsilon Greedy", "UCB"]
        agents = [greedy, e_greedy, ucb]
        rewards = [0 for _ in range(len(agents))]
        for i in range(steps):
            for j in range(len(agents)):
                action = agents[j].choose_action()
                reward = bandit.step(action)
                agents[j].update(action, reward)
                rewards[j] += reward
        print("Param Value: {}".format(v))
        print("Average Rewards:")
        for i in range(len(agents)):
            print("{} Agent: {}".format(names[i], rewards[i]/steps))
        print()
    print(param_vals)

if __name__ == "__main__":
    main()
