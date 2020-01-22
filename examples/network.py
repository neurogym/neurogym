#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 07:27:54 2020

@author: manuel

Update-Gate RNN (Collins et al. 2017).
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils as ut

def RNN_UGRU(inputs, prev_rewards, a_size, num_units):

    # create a UGRNNCell
    rnn_cell = tf.contrib.rnn.UGRNNCell(num_units, activation=tf.nn.relu)

    # this is the initial state used in the A3C model when training
    # or obtaining an action
    st_init = np.zeros((1, rnn_cell.state_size), np.float32)

    # defining initial state
    state_in = tf.placeholder(tf.float32, [1, rnn_cell.state_size])

    # reshape inputs size
    rnn_in = tf.expand_dims(inputs, [0])

    step_size = tf.shape(prev_rewards)[:1]

    # 'state' is a tensor of shape [batch_size, cell_state_size]
    # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
    outputs, state_out = tf.nn.dynamic_rnn(rnn_cell, rnn_in,
                                           initial_state=state_in,
                                           sequence_length=step_size,
                                           dtype=tf.float32,
                                           time_major=False)

    rnn_out = tf.reshape(outputs, [-1, num_units])

    actions, actions_onehot, policy, value = \
        process_output(rnn_out, outputs, a_size, num_units)

    return st_init, state_in, state_out, actions, actions_onehot, policy, value


def process_output(rnn_out, outputs, a_size, num_units):
    # Actions
    actions = tf.placeholder(shape=[None], dtype=tf.int32)
    actions_onehot = tf.one_hot(actions, a_size, dtype=tf.float32)

    # Output layers for policy and value estimations
    policy = slim.fully_connected(rnn_out, a_size,
                                  activation_fn=tf.nn.softmax,
                                  weights_initializer=ut.normalized_columns_initializer(0.01),
                                  biases_initializer=None)
    value = slim.fully_connected(rnn_out, 1,
                                 activation_fn=None,
                                 weights_initializer=ut.normalized_columns_initializer(1.0),
                                 biases_initializer=None)

    return actions, actions_onehot, policy, value
