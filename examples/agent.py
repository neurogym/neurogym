#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 07:29:40 2020

@author: manuel
"""
import tensorflow as tf
import network as net
import tensorflow.contrib.slim as slim
import utils as ut
import numpy as np
import matplotlib.pyplot as plt


class AC_Network():
    def __init__(self, a_size, state_size, scope, trainer, num_units):
        with tf.variable_scope(scope):
            # Input
            self.st = tf.placeholder(shape=[None, 1, state_size, 1],
                                     dtype=tf.float32)
            self.prev_rewards = tf.placeholder(shape=[None, 1],
                                               dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None],
                                               dtype=tf.int32)

            self.prev_actions_onehot = tf.one_hot(self.prev_actions, a_size,
                                                  dtype=tf.float32)

            hidden = tf.concat([slim.flatten(self.st), self.prev_rewards,
                                self.prev_actions_onehot], 1)

            self.st_init, self.st_in, self.st_out, self.actions,\
                self.actions_onehot, self.policy, self.value =\
                net.RNN_UGRU(hidden, self.prev_rewards, a_size, num_units)

            # Only the worker network needs ops for loss functions
            # and gradient updating.
            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],
                                                 dtype=tf.float32)

                self.resp_outputs = \
                    tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(
                    tf.square(self.target_v -
                              tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(
                    self.policy * tf.log(self.policy + 1e-7))
                self.policy_loss = -tf.reduce_sum(
                    tf.log(self.resp_outputs + 1e-7)*self.advantages)
                self.loss = 0.5 * self.value_loss +\
                    self.policy_loss -\
                    self.entropy * 0.05

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms =\
                    tf.clip_by_global_norm(self.gradients, 999.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars))


class Worker():
    def __init__(self, game, name, a_size, state_size, trainer,
                 global_epss, num_units):
        self.name = "worker_" + str(name)
        self.number = name
        self.trainer = trainer
        self.global_epss = global_epss
        self.increment = self.global_epss.assign_add(1)
        self.eps_rewards = []
        self.eps_mean_values = []

        # Create the local copy of the network and the tensorflow op
        # to copy global parameters to local network
        self.local_AC = AC_Network(a_size, state_size, self.name, trainer,
                                   num_units)
        self.update_local_ops = ut.update_target_graph('global', self.name)
        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]

        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:, 3]

        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = ut.discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards +\
            gamma * self.value_plus[1:] -\
            self.value_plus[:-1]
        advantages = ut.discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.st_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.st: np.stack(states, axis=0),
                     self.local_AC.prev_rewards: np.vstack(prev_rewards),
                     self.local_AC.prev_actions: prev_actions,
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.st_in: rnn_state}

        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        aux = len(rollout)
        return v_l / aux, p_l / aux, e_l / aux, g_n, v_n

    def work(self, gamma, sess, coord, saver, train):
        eps_count = sess.run(self.global_epss)
        num_epss_end = 100000
        num_tr_update = 1000
        total_steps = 0
        print("Starting worker " + str(self.number))
        # create performance figure
        if self.number == 0:
            fig_perf = plt.figure(figsize=(8, 8), dpi=100)
            perf_mat = []
        # get first state
        s = self.env.reset()
        s = np.reshape(s, [1, s.shape[0], 1])
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                eps_buffer = []
                eps_values = []
                eps_reward = 0
                outcome = []
                eps_step_count = 0
                num_tr = 0
                update = False
                r = 0
                a = 0
                rnn_state = self.local_AC.st_init
                while not update:
                    feed_dict = {self.local_AC.st: [s],
                                 self.local_AC.prev_rewards: [[r]],
                                 self.local_AC.prev_actions: [a],
                                 self.local_AC.st_in: rnn_state}

                    # Take an action using probs from policy network output
                    a_dist, v, rnn_state_new = sess.run(
                                                        [self.local_AC.policy,
                                                         self.local_AC.value,
                                                         self.local_AC.st_out],
                                                        feed_dict=feed_dict)

                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    rnn_state = rnn_state_new
                    # new_state, reward, update_net, new_trial
                    s1, r, d, info = self.env.step(a)
                    s1 = np.reshape(s1, [1, s1.shape[0], 1])
                    # save samples for training the network later
                    eps_buffer.append([s, a, r, v[0, 0]])
                    eps_values.append(v[0, 0])
                    eps_reward += r
                    total_steps += 1
                    eps_step_count += 1
                    s = s1
                    if info['new_trial']:
                        outcome.append(r)
                        num_tr += 1
                    if (num_tr+1) % num_tr_update == 0 or d:
                        update = True

                # Update the network using the experience buffer
                # at the end of the episode
                if len(eps_buffer) != 0 and train:
                    v_l, p_l, e_l, g_n, v_n = \
                        self.train(eps_buffer, sess, gamma, 0.0)
                    if self.number == 0:
                        mean_perf = np.mean(outcome)
                        perf_mat.append(mean_perf)
                        fig_perf.plot((np.arange(len(mean_perf))+1)*num_tr_update, perf_mat)
                        fig_perf.canvas.draw()
                        plt.title('average performance: ' + str(mean_perf))

                if self.name == 'worker_0':
                    sess.run(self.increment)

                eps_count += 1
                if eps_count > num_epss_end:
                    break
