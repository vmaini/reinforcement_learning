# Source: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()
render = False # set to True to visualize the game

# one-layer network
#
# X [1,16] -- input, one-hot state vector
# W [16,4]  B [4]
# Q_out [1x4] -- Q-values for each output

X = tf.placeholder(shape = [1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Q_out = tf.matmul(X,W)
predict = tf.argmax(Q_out, axis=1)

# calculate loss
Q_next = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Q_next-Q_out))
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train_step = trainer.minimize(loss)

# hyperparameters
discount_rate = 0.99
epsilon = 0.1
num_episodes = 2000

j_list = []
rewards_list = []
running_reward = 0
running_rewards = []

# train the network

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in xrange(num_episodes):
    s = env.reset()
    reward_sum = 0
    done = False
    j = 0

    while j < 99:
        if render: env.render()
        j += 1
        # choose action greedily, with epsilon chance of random action
        a, Q_all = sess.run([predict, Q_out],feed_dict={X:np.identity(16)[s:s+1]}) # passes in one-hot state vector as input

        if np.random.rand(1) < epsilon:
            a[0] = env.action_space.sample()

        s1, reward, done, info = env.step(a[0])

        # get max(Q'(s',a')) from Bellman equation to set our target value for chosen action
        Q1 = sess.run(Q_out, feed_dict={X:np.identity(16)[s1:s1+1]})
        max_Q1 = np.max(Q1)
        Q_target = Q_all
        Q_target[0,a[0]] = reward + discount_rate * max_Q1

        # train network with target & %dpr % iediction, update reward sum & state

        info, W1 = sess.run([train_step,W],feed_dict={X:np.identity(16)[s:s+1],Q_next:Q_target})
        reward_sum += reward
        s = s1

        if done:
            epsilon = 1./((i/50)+10)
            break

    j_list.append(j)
    rewards_list.append(reward_sum)
    running_reward += reward_sum
    running_rewards.append(running_reward)

    if i % 100 == 0:
        print "epoch %d reward: %d" % (i/100, running_reward)
        running_reward = 0

print "average success rate: " + str(sum(rewards_list)/num_episodes * 100) + "%"
print "best epoch: " + str(max(running_rewards) * 100) + "%"
