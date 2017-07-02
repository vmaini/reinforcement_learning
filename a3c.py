# sources: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
# https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/
# https://github.com/openai/universe-starter-agent

'''
A3C ALGORITHM

Global network: input (s) --> network --> pi(s), v(s)
Agent instances 1..N

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

from helper import *
from vizdoom import *


'''
- process the state input
- define ConvNet architecture for
'''
class Brain():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # input layer
            self.inputs = tf.placeholder(shape=[None,s_size], dtype=tf.float32)
            
