# source: https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-3-model-based-rl-9a6fe0cce99

import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import cPickle as pickle

env = gym.make('CartPole-v0')
resume = False
render = False

def reset_grad_buffer(grad_buffer):
    for idx, grad in enumerate(grad_buffer):
        grad_buffer[idx] = grad * 0
    return grad_buffer

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def step_model(sess,xs,action):
    to_feed = np.reshape(np.hstack([xs[-1][0],np.array(action)]),[1,5])
    prediction = sess.run([predicted_state],feed_dict={previous_state: to_feed})
    reward = prediction[0][:,4]
    observation = prediction[0][:,0:4]
    observation[:,0] = np.clip(observation[:,0],-2.4,2.4)
    observation[:,2] = np.clip(observation[:,2],-0.4,0.4)
    done_p = np.clip(prediction[0][:,5],0,1)
    if done_p > 0.1 or len(xs)>= 300:
        done = True
    else:
        done = False
    return observation, reward, done

# hyperparameters
D = 4 # input dimensions
H = 8 # hidden layer size

lr = 1e-2
gamma = 0.99 # discount rate
decay_rate = 0.99

model_batch_size = 3
real_batch_size = 3


'''
POLICY NETWORK
- Set up two-layer fully-connected network architecture

X [N, D] -- input observation
W1 [D, H]
Y1 [N, H] -- fully connected layer, relu activation
W2 [H,1]
logp [N,1] -- fully connected layer, outputs logp(action)

- Compute loss
- Define optimizer
- Define weight update procedure
'''

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None,D], name="X") # input observation

W1 = tf.get_variable("W1", shape=[D,H], initializer=tf.contrib.layers.xavier_initializer()) # creates trainable var holding [DxH] weights with Xavier initialization
Y1 = tf.nn.relu(tf.matmul(X,W1)) #

W2 = tf.get_variable("W2", shape=[H,1],initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(Y1, W2)
Y = tf.nn.sigmoid(score) # probability

tvars = tf.trainable_variables()
Y_ = tf.placeholder(tf.float32, [None,1], name="Y_")

advantages = tf.placeholder(tf.float32, name="advantages")
optimizer = tf.train.AdamOptimizer(learning_rate = lr)

dW1 = tf.placeholder(tf.float32,name="batch_grad1")
dW2 = tf.placeholder(tf.float32,name="batch_grad2")

batch_grad = [dW1, dW2]

# compute loss: log(π)*A
log_likelihood = tf.log(Y_ * (Y_ - Y) + (1-Y_) * (Y_ + Y))
loss = -tf.reduce_mean(log_likelihood * advantages)

# store the new gradient
new_grads = tf.gradients(loss, tvars)

# apply gradient to update weights
update_grads = optimizer.apply_gradients(zip(batch_grad, tvars))

'''
MODEL NETWORK
- Set up RNN architecture

INPUT: current state, action
OUTPUT: observation, reward, done state

- Calculate loss
- Define optimizer
- Define weight update procedure
'''
Hm = 256 # model network hidden layer size
input_data = tf.placeholder(tf.float32, [None, 5])
xavier = tf.contrib.layers.xavier_initializer()

with tf.variable_scope('rnnlm'): # allows for shared variables in more complex models
# see: https://www.tensorflow.org/programmers_guide/variable_scope
    softmax_w = tf.get_variable("softmax_w",[Hm,50])
    softmax_b = tf.get_variable("softmax_b",[50])

previous_state = tf.placeholder(tf.float32, [None,5], name="previous_state")

W1m = tf.get_variable("W1m",shape=[5,Hm],initializer=xavier)
B1m = tf.Variable(tf.zeros([Hm]),name="B1m")
Y1m = tf.nn.relu(tf.matmul(previous_state,W1m)+B1m)

W2m = tf.get_variable("W2m",shape=[Hm, Hm], initializer=xavier)
B1m =tf.Variable(tf.zeros([Hm]),name="B2m")
Y2m = tf.nn.relu(tf.matmul(Y1m,W2m)+B1m)

W_obs = tf.get_variable("W_obs",shape=[Hm,4],initializer=xavier)
W_reward = tf.get_variable("W_reward",shape=[Hm,1],initializer=xavier)
W_done = tf.get_variable("W_done",shape=[Hm,1], initializer=xavier)

B_obs = tf.Variable(tf.zeros([4]),name="B_obs")
B_reward = tf.Variable(tf.zeros([1]),name="B_reward")
B_done = tf.Variable(tf.zeros([1]),name="B_done")

predicted_obs = tf.matmul(Y2m, W_obs, name="predicted_obs") + B_obs
predicted_reward = tf.matmul(Y2m, W_reward, name="predicted_reward") + B_reward
predicted_done = tf.sigmoid(tf.matmul(Y2m,W_done,name="predicted_done")+B_done)

true_obs = tf.placeholder(tf.float32,[None,4],name="true_obs")
true_reward = tf.placeholder(tf.float32,[None,1],name="true_reward")
true_done = tf.placeholder(tf.float32,[None,1],name="true_done")

predicted_state = tf.concat([predicted_obs,predicted_reward, predicted_done],1)

observation_loss = tf.square(true_obs - predicted_obs)
reward_loss = tf.square(true_reward - predicted_reward)
done_loss = tf.multiply(predicted_done, true_done) + tf.multiply(1-predicted_done, 1-true_done) #cross-entropy loss
done_loss = -tf.log(done_loss)

model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)

model_optimizer = tf.train.AdamOptimizer(learning_rate = lr)
update_model = model_optimizer.minimize(model_loss)

'''
TRAINING
'''

xs, drs, ys, ds = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
batch_size = real_batch_size

#settings for model observation vs. real environment observation, model training vs. policy training
draw_from_model = False
train_model = True
train_policy = False
switch_point = 1

render = False

# graph execution
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

observation = env.reset()
x = observation
grad_buffer = sess.run(tvars) # remember, tvars = weights
grad_buffer = reset_grad_buffer(grad_buffer)

while episode_number <= 5000:
    if render: env.render()

    x = np.reshape(observation, [1,4])

    # choose action
    tfprob = sess.run(Y,feed_dict={X:x})
    action = 1 if np.random.uniform() < tfprob else 0

    # record intermediates for backrpop
    xs.append(x)
    y = 1 if action == 0 else 0
    ys.append(y)

    # model step / real env step
    if draw_from_model == False:
        observation, reward, done, info = env.step(action)
    else:
        observation, reward, done = step_model(sess,xs,action)

    reward_sum += reward

    ds.append(done*1)
    drs.append(reward)

    if done:
        if draw_from_model == False:
            real_episodes += 1
        episode_number += 1

    # stack inputs, hidden states, action gradients, rewards

        epx = np.vstack(xs)
        epy = np.vstack(ys)
        epr = np.vstack(drs)
        epd = np.vstack(ds)

        # reset memory
        xs, drs, ys, ds = [],[],[],[]

        if train_model:
            actions = np.array([np.abs(y-1) for y in epy][:-1])
            state_prevs = epx[:-1,:]
            state_prevs = np.hstack([state_prevs,actions])
            state_nexts = epx[1:,:]
            rewards = np.array(epr[1:,:])
            dones = np.array(epd[1:,:])
            state_nextsAll = np.hstack([state_nexts,rewards,dones])

            feed_dict={previous_state: state_prevs, true_obs: state_nexts,true_done:dones,true_reward:rewards}
            loss,pState,_ = sess.run([model_loss,predicted_state,update_model],feed_dict)

        if train_policy:
            discounted_epr = discount_rewards(epr).astype('float32')
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            tGrad = sess.run(new_grads,feed_dict={X: epx, Y_: epy, advantages: discounted_epr})

             # If gradients becom too large, end training process
            if np.sum(tGrad[0] == tGrad[0]) == 0:
                break
            for ix,grad in enumerate(tGrad):
                grad_buffer[ix] += grad

        if switch_point + batch_size == episode_number:
            switch_point = episode_number
            if train_policy:
                sess.run(update_grads,feed_dict={dW1: grad_buffer[0],dW2:grad_buffer[1]})
                grad_buffer = reset_grad_buffer(grad_buffer)

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

        if draw_from_model == False:
            print('World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' % (real_episodes,reward_sum/real_batch_size,action, running_reward/real_batch_size))
            if reward_sum/batch_size > 200:
                break
        reward_sum = 0

        # after 100 episodes, alternate between training policy from model and real env
        if episode_number > 100:
            draw_from_model = not draw_from_model
            train_model = not train_model
            train_policy = not train_policy

        if draw_from_model == True:
            observation = np.random.uniform(-0.1,0.1,[4]) # Generate reasonable starting point
            batch_size = model_batch_size
        else:
            observation = env.reset()
            batch_size = real_batch_size

print real_episodes
