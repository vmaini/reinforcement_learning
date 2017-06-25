# Reinforcement learner based on Andrej Karpathy's Pong from Pixels: http://karpathy.github.io/2016/05/31/rl/
# and Dhruv Parthasarathy's implementation: https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0

# TODO: add performance visualization graph over time

import gym
import numpy as np
import cPickle as pickle

'''
Architecture
- raw_input = preprocessed pixel data [6400 x 1]
- W1 [200 x 6400] * input [6400 x 1] --> hidden layer [200 x 1]
- W2 [1 x 200] * hidden layer [200 x 1] --> output P [1 x 1]

Forward pass: play round
- preprocess pixel data X: frame-to-frame delta, remove color, downsample
- compute hidden layer H --> ReLU activation function --> P = sigmoid(output)
- finish round --> determine reward (win/lose)

Backprop: compute gradient
- after every episode (someone got 21 points), pass results through backprop & compute weight gradient
    * dC_dw2
    * delta_1
    * dC_dW1

Weight updates
- after 10 episodes, sum gradient and move weights in direction of the gradient

Repeat until our player does better than NPC
'''

# hyperparameters
H = 200 # number of hidden layer neurons
D = 80*80 # pixel dimensions of game
batch_size = 10 # num episodes per weight update
gamma = .99 # discount rate reward
decay_rate = .99 # decay factor for RMSProp
learning_rate = 1e-4
resume = True # resume from previous checkpoint?
render = True # set to True to watch the game

# initialize weights
if resume:
    weights = pickle.load(open('save.p','rb'))
else:
    # Xavier weight initialization
    weights = {
    'W1': np.random.randn(H,D) / np.sqrt(D),
    'W2': np.random.randn(H) / np.sqrt(H)
    }

# RMSProp setup
gradient_buffer = {}
rmsprop_cache = {}
for k, v in weights.iteritems():
    gradient_buffer[k] = np.zeros_like(v) # will keep sum of gradients in a batch
    rmsprop_cache[k] = np.zeros_like(v) # initialize RMSprop memory

def preprocess(raw_obs, prev_processed_obs):
    '''
    INPUT:
    - raw_obs: raw input, [80x80] matrix of pixel values
    - prev_processed_obs: previous processed observation, [6400x1] float vector

    OUTPUT:
    - diff_frame: change from previous to current processed frame, [6400x1] float vector
    - prev_processed_obs: stores current processed obs as previous for next iteration, [6400x1] float vector
    '''
    # process the observation
    processed_obs = raw_obs[35:195] # crop
    processed_obs = processed_obs[::2, ::2, :] # downsample (1/2 resolution)
    processed_obs = processed_obs[:,:,0] # remove color <-- 0th element of RGB dimension
    processed_obs[processed_obs!=0] = 1 # set non-black things, e.g. paddles/ball, = 1
    processed_obs[processed_obs==144] = 0 # remove background
    processed_obs[processed_obs==109] = 0 # remove background
    processed_obs = processed_obs.astype(np.float).ravel() # 80x80 --> 1600x1 float column vector


    # compute difference frame (zeros if obs = first frame of round)
    if prev_processed_obs is not None:
        diff_frame = processed_obs - prev_processed_obs
    else:
        diff_frame = np.zeros_like(processed_obs)
        diff_frame = diff_frame.astype(np.float).ravel()

    # store current processed obs as previous for next time
    prev_processed_obs = processed_obs
    return diff_frame, prev_processed_obs

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def policy_forward(x):
    '''
    INPUT:
    - x: processed difference frame as network input [6400x1]

    OUTPUT:
    - hidden layer values: [200x1] float vector
    - probability of moving up: float in [0,1]
    '''
    h = np.dot(weights['W1'],x)
    h[h<0] = 0 #ReLU
    logp = np.dot(weights['W2'], h) #calculates log probability of going up

    return h, sigmoid(logp)

def choose_action(prob):
    '''
    INPUT:
    - prob: P(up)

    OUTPUT:
    - 2 or 3, i.e. UP or DOWN (Gym presets)
    '''
    return 2 if np.random.uniform() < prob else 3

def discount_rewards(r):
    '''
    INPUT:
    - 1D numpy array of rewards

    OUTPUT:
    - 1D numpy array of discounted rewards
    '''
    discounted_r = np.zeros_like(r)
    discounted_sum = 0

    for t in reversed(xrange(0, r.size)):
        discounted_sum = discounted_sum * gamma + r[t]
        discounted_r[t] = discounted_sum
    return discounted_r


def policy_backward(episode_hs, episode_dlogps):
    '''calculate gradient. walkthrough: http://neuralnetworksanddeeplearning.com/chap2.html '''

    dW2 = np.dot(episode_hs.T, episode_dlogps).ravel()
    dh = np.outer(episode_dlogps, weights['W2'])
    dh[episode_hs<=0] = 0 # relu
    dW1 = np.dot(dh.T, episode_xs)
    gradient = {'W1':dW1, 'W2':dW2}
    return gradient

def update_weights(weights, gradient_buffer, rmsprop_cache, decay_rate, learning_rate):
    epsilon = 1e-5

    for k,v in weights.iteritems():
        g = gradient_buffer[k] #gradient sum
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1-decay_rate) * g**2
        weights[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        gradient_buffer[k] = np.zeros_like(v)

# set up the environment
env = gym.make("Pong-v0")
observation = env.reset()

# store episode information
prev_x = None # stores previous obs, for computing difference frame
episode_number = 0
reward_sum = 0
running_reward = None
xs, hs, dlogps, rewards = [], [], [], [] # stores observations, hidden layer values, logp gradients on loss function, rewards

# play and learn loop
while True:
    if render: env.render() # show gameplay

    # get preprocessed difference frame x, store current x as previous x
    x, prev_x = preprocess(observation, prev_x)

    # forward pass, choose an action
    h, p_up = policy_forward(x)
    action = choose_action(p_up)

    # store intermediate observations & hidden layer values for backprop later
    xs.append(x)
    hs.append(h)

    # map actions to 0s/1s as makeshift "labels", as in supervised learning
    y = 1 if action == 2 else 0
    dlogps.append(y - p_up) # gradient on loss w.r.t. parameters. positive gradient means: increase parameter --> logp(y=action_taken|x) increases. we'll encourage the action that was taken and check if it was a good action later

    # carry out the action, step the environment, record reward for the action
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    rewards.append(reward)

    '''
    After each episode (player reaches 21), perform parameter updates based on the combined results of each round:
    - stack xs, hs, action gradients (dlogp(up)), and rewards for the episode
    - compute discounted rewards, going backwards
    - perform RMSProp after each batch of episodes
    '''
    if done:
        episode_number += 1

        # combine episode inputs, hidden layer values, action gradients, and rewards. we'll use this for updating the policy

        episode_xs = np.vstack(xs) # stack processed observation row vectors vertically
        episode_hs = np.vstack(hs) # combine intermediate hidden layer values
        episode_dlogps = np.vstack(dlogps)
        episode_rewards = np.vstack(rewards)

        # compute normalized discounted rewards
        discounted_episode_rewards = discount_rewards(episode_rewards)
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)

        # modulate the gradient with "advantage": if the action yielded positive reward, policy gradient will increase logp(action), else will discourage the action.
        episode_dlogps *= discounted_episode_rewards

        gradient = policy_backward(episode_hs, episode_dlogps)

        # keep sum of gradients throughout the batch for RMSProp
        for k in weights:
            gradient_buffer[k] += gradient[k]

        # perform RMSProp every batch_size episodes: keep a moving average of the squared gradient for each weight, divide gradient by sqrt(meansquare(w,t)). see: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        if episode_number % batch_size == 0:
            update_weights(weights,gradient_buffer,rmsprop_cache,decay_rate,learning_rate)

        # keep track of things as training runs
        running_reward = reward_sum if running_reward is None else running_reward * .99 + reward_sum * .01
        print 'resetting env. episode reward total: %f. running mean: %f' % (reward_sum, running_reward)
        if episode_number % 100 == 0:
            pickle.dump(weights, open('save.p', 'wb'))
            print 'saving weights'

        #reset everything
        xs, hs, dlogps, rewards = [],[],[],[]
        reward_sum = 0
        prev_x = None
        observation = env.reset()

    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
      print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
