# Source: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('FrozenLake-v0')

# initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros([num_states, num_actions])

# hyperparameters
learning_rate = .8
discount_rate = .95
num_episodes = 2000

rewards_list = []
running_reward = 0
running_rewards = []

for i in range(num_episodes):
    s = env.reset() # state in Q(s,a)
    reward_sum = 0
    done = False
    j = 0

    while j < 99:
        j += 1
        # greedily pick the best action (column) given the state (row), add noise

        a = np.argmax(Q[s,:] + np.random.randn(1,num_actions)*(1./(i+1)))
         #action in Q(s,a)

        s_prime, reward, done, info = env.step(a)

        # update Q-table to incorporate results of previous action (add diff between Q(s',a) and Q(s,a) modulated by learning rate)
        Q[s,a] = Q[s,a] + learning_rate * (reward + discount_rate * np.max(Q[s_prime,:]) - Q[s,a])
        reward_sum += reward
        s = s_prime

        if done:
            break
    rewards_list.append(reward_sum)
    running_reward += reward_sum

    if i % 100 == 0:
        running_rewards.append(running_reward)
        print "epoch %d reward: %d" % (i/100, running_reward)
        running_reward = 0

plt.hist(rewards_list, label = "num episodes", facecolor='g', alpha=.6)
plt.title('Q-table performance: FrozenLake-v0')
plt.xlabel('reward')
plt.legend()
plt.show()

print "average success rate: " + str(sum(rewards_list)/num_episodes * 100) + "%"
print "best epoch: " + str(max(running_rewards)) + "%"

print 'final Q-table values:'
print Q
