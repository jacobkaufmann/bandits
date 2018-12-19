import numpy as np
import math 

from bandit import Bandit
import agent

def main():
    k = 10
    alpha = 0.1
    runs = 50
    steps = 1000

    mean = 0
    variance = 1
    nonstationary = False

    param_vals = [2**i for i in range(-7, 3)]
    names = ["Greedy", "Epsilon Greedy", "UCB", "Gradient"]
    N = len(names)
    avgs = []

    for v in param_vals:
        rewards = [0 for _ in range(N)]

        for _ in range(runs):
            # K-armed bandit problem
            bandit = Bandit(k, mean, variance, nonstationary)

            # Agents
            greedy = agent.GreedyAgent(k, alpha=alpha, optimism=v)
            e_greedy = agent.EpsilonGreedyAgent(k, alpha=alpha, epsilon=v)
            ucb = agent.UcbAgent(k, alpha=alpha, c=v)
            grad = agent.GradientAgent(k, alpha=v)
            agents = [greedy, e_greedy, ucb, grad]

            for i in range(steps):
                for j in range(N):
                    action = agents[j].choose_action()
                    reward = bandit.step(action)
                    agents[j].update(action, reward)
                    rewards[j] += reward

        avgs.append([rewards[i]/(runs*steps) for i in range(N)])

        print("Param Value: {}".format(v))
        print("Average Rewards:")
        for i in range(N):
            print("\t- {} Agent: {:.4f}".format(names[i], avgs[-1][i]))
        print()

    bests = np.argmax(avgs, axis=0)
    for i in range(N):
        print("Optimal Param Value for {}: {}".format(names[i], param_vals[bests[i]]))

if __name__ == "__main__":
    main()
