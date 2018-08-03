import numpy as np
import random
import math

def ind_max(x):
  m = max(x)
  return x.index(m)

# def test_algorithm(algo, arms, num_sims, horizon):
#     chosen_arms = [0.0 for i in range(num_sims * horizon)]
#     rewards = [0.0 for i in range(num_sims * horizon)]
#     cumulative_rewards = [0.0 for i in range(num_sims * horizon)]
#     sim_nums = [0.0 for i in range(num_sims * horizon)]
#     times = [0.0 for i in range(num_sims * horizon)]
#
#     for sim in range(num_sims):
#         sim = sim + 1
#
#         for t in range(horizon):
#             t = t + 1
#             index = (sim - 1) * horizon + t - 1
#             sim_nums[index] = sim
#             times[index] = t
#
#             chosen_arm = algo.select_arm()
#             chosen_arms[index] = chosen_arm
#
#             reward = arms[chosen_arms[index]].draw()
#             rewards[index] = reward
#
#             if t == 1:
#                 cumulative_rewards[index] = reward
#             else:
#                 cumulative_rewards[index] = cumulative_rewards[index - 1] + reward
#
#             algo.update(chosen_arm, reward)
#
#     return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]

class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0

class UCB :
    def __init__(self, N, e):
        # Nt : count of each state, t : total counts
        self.Nt = np.zeros(N) + e
        self.t = np.sum(self.Nt)

    def get_U(self, n):
        print(self.t, self.Nt)
        a = (2 * math.log(self.t))
        U = math.sqrt(a / float(self.Nt[n]))
        return U

    def update(self, n):
        self.Nt[n] += 1
        self.t += 1


# execfile("core.py")

random.seed()
# means = [0.1, 0.1, 0.1, 0.1, 0.9]
# n_arms = len(means)
# random.shuffle(means)
# arms = map(lambda (mu): BernoulliArm(mu), means)
# print("Best arm is " + str(ind_max(means)))

# n_arms = 5
epsilon = 1
prob = [0, 0.1, 0.2, 0.4, 0.6, 1]
n_arms = len(prob)-1
ucb = UCB(n_arms, epsilon)
for i in range(1000) :
    nexon = random.random()
    for j in range(n_arms) :
        if prob[j] <= nexon < prob[j+1] :
            n = j
    U = ucb.get_U(n)
    if i % 10 == 0 :
        print(U, n)
    ucb.update(n)

# results = test_algorithm(ucb, arms, 5000, 250)

# f = open("algorithms/ucb/ucb1_results.tsv", "w")
#
# for i in range(len(results[0])):
#   f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
#
# f.close()