# sources: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
# https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/
# https://github.com/openai/universe-starter-agent

'''
A3C ALGORITHM

Global network: input (s) --> network --> pi(s), v(s)
Agent instances

Asynchronous: multiple agents with independent copies of environment --> uncorrelated experiences
Actor-critic: V(s), pi(s)
Advantage: A = R - V(s)

'''

import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal

from a3c_helper import *
from vizdoom import *


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # input layer
            self.inputs = tf.placeholder(shape=[None,s_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,81,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.imageIn, num_outputs = 16, kernel_size=[8,8],stride=[4,4],stride=2,2],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn = tf.nn.elu,inputs=self.conv1,num_outputs=32,kernel_size=[4,4],stride=[2,2],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)

            # RNN for temporal dependencies
            # more on LSTMs: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
            c_init = np.zeros((1,lstm_cell.state_size.c),np.float32)
            h_init = np.zeros((1,lstm_cell.state_size.h),np.float32)
            self.state_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, [1,lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32,[1,lstm_cell.state_size.h])
            self.state_in = (c_in, h_in) #state is tuple
            rnn_in = tf.expand_dims(hidden,[0]) #inserts a dimension of 1 at dimension index 0
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state = state_in, sequence_length = step_size, time_major = False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # policy and value network outputs
             self.policy = slim.fully_connected(rnn_out, a_size, activation_fn = tf.nn.softmax, weights_initializer=normalized_columns_initializer(0.01))

             self.value = slim.fully_connected(rnn_out,1,activation_fun=None,weights_initializer=normalized_columns_initializer(1.0),biases_initializer=None)


        #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.gobal_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lenghts = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str.self.number))

        # creaate local copy of network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        # set up DOOM

        game.set_doom_scenario_path("basaic.wad")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_render_hud(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()

        self.actions = self.actions = np.identity(a_size,dtype=bool).tolist()
        self.env = game

    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]

        # calculate advantage & discounted returns from rollout

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma(self.value_plus[1:] - self.value_plus[:-1])
        advantages = discount(advantages, gamma)

        # update global network with gradients

        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.taarget_v: discounted_rewards, self.locaal_AC.inputs:np.vstack(observations), self.local_AC.advantages:advantages, self.local_AC.state_in[0]:rnn_state[0], self.local_AC.state_in[1]:rnn_state[1]}

        v_l, p_l, e_l, g_n, v_n = sess.run([self.local_AC.value_loss, self.local_AC.policy_loss, self.local_AC.entropy, self.local_AC.grad_norms, self.local_AC.var_norms, self.local_AAC.apply_grads], feed_dict = feed_dict)

        return v_l / len(rollout), p_1 / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver):
        # plays in the environment
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("starting worker " + str(self.number))

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                self.env.new_episode()
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_AC.state_init

                while self.env.is_episode_finished() == False:
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], feed_dict={self.local_AC.inputs:[s], self.local_AC.state_in[0]:rnn_state[0], self.local_AC.state_in[1]:rnn_state[1]})

                    # choose action from policy network output
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    r = self.env.make_action(self.actions[a]) / 100.0
                    d = self.env.is_episode_finished()

                    if d == False:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1)

                    else:
                        s1 = s

                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # if experience buffer is full, make an updaate using that experience rollout
                    # final return is unknown, so bootstrap from current value estimation
                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length -1:
                        v1 = sess.run(self.local_AC.value, feed_dict={self.local_AC.inputs:[s], self.local_AC.state_in[0]:rnn_state[0], self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # update network using experience buffer at end of episode

                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)


                # bookkeeping: save gifs of episodes, model params, and summary stats
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step,true_image=True,salience=False)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
a_size = 3 # Agent can move Left, Right, or Fire
load_model = False
model_path = './model'


tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

#Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(DoomGame(),i,s_size,a_size,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
