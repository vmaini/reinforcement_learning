# source: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
# https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib as plt

env = gym.make('CartPole-v0')
render = True

gamma = .99

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # feed-forward architecture for determining state from action

        self.state_in = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn = tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.action=tf.argmax(self.output,1)

        # training: feed reward and action into network, compute loss, update network
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indices = tf.range(0, tf.shape(self.output)[0])*tf.shape(self.output)[1] + self.action
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indices)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward)

        tvars = tf.trainable_variables()
        self.gradient_holders = []

        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

tf.reset_default_graph()

player = agent(lr=1e-2,s_size=4,a_size=2,h_size=8)

total_episodes = 2000
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        s = env.reset()
        if render: env.render()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            # pick an action given outputs
            a_dist = sess.run(player.output,feed_dict={player.state_in:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)

            s1,r,d,_ = env.step(a)
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r
            if d == True:
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                feed_dict={player.reward:ep_history[:,2],
                        player.action:ep_history[:,1],player.state_in:np.vstack(ep_history[:,0])}
                grads = sess.run(player.gradients, feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict= dictionary = dict(zip(player.gradient_holders, gradBuffer))
                    _ = sess.run(player.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                total_length.append(j)
                break

        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1
