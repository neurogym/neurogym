#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 08:52:21 2019

@author: molano
"""

import gym
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt


class ngym(gym.Env):
    """
    two-alternative forced choice task where the probability of repeating the
    previous choice is parametrized
    """
    def __init__(self, dt=0.1):
        super().__init__()
        self.dt = dt  # TODO: revisit, ms or s?
        self.t = 0
        self.num_tr = 0
        self.rng = np.random.RandomState(seed=0)
        self.perf = 0
        self.num_tr_perf = 0
        self.num_tr_exp = 10000  # num trials after which done = True
        # for rendering
        self.obs_mat = []
        self.act_mat = []
        self.rew_mat = []
        self.new_tr_mat = []
        self.max_num_samples = 100
        self.num_subplots = 3
        self.fig, self.ax = plt.subplots(self.num_subplots, 1)

    def step(self, action):
        """
        receives an action and returns a new state, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information
        """
        return None, None, None, None

    def reset(self):
        """
        restarts the experiment with the same parameters
        """
        if len(self.rew_mat) > 0:
            print('percentage of trials performed: ' +
                  str(100*self.num_tr_perf/self.num_tr_exp))
            print('mean performance: ' + str(self.perf))
            self.render()

        self.perf = 0
        self.num_tr_perf = 0
        self.num_tr = 1
        self.t = 0
        self.obs_mat = []
        self.act_mat = []
        self.rew_mat = []
        self.trial = self._new_trial()
        obs, _, _, _ = self.step(self.action_space.sample())
        return obs

    def render(self, mode='human'):
        """
        plots relevant variables/parameters
        """
        # observations
        obs = np.array(self.obs_mat)
        self.ax[0].imshow(obs.T, aspect='auto')
        self.ax[0].set_ylabel('obs')
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])
        # actions
        self.ax[1].plot(np.arange(1, len(self.act_mat)+1)+0.5, self.act_mat)
        self.ax[1].set_ylabel('action')
        self.ax[1].set_xlim([0, self.max_num_samples+0.5])
        self.ax[1].set_xticks([])
        self.ax[1].set_yticks([])
        # reward
        self.ax[2].plot(np.arange(1, len(self.act_mat)+1)+0.5, self.rew_mat)
        self.ax[2].set_ylabel('reward')
        self.ax[2].set_xlim([0, self.max_num_samples+0.5])
        self.fig.savefig('trials.svg')
        plt.cla()

    # Auxiliary functions
    def seed(self, seed=None):  # TODO: what is this function for?
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _new_trial(self, rng, dt):
        """
        starts a new trials within the current experiment
        """
        pass

    def in_epoch(self, t, epoch):
        """Check if t is in epoch."""
        dur = self.trial['durations']
        return dur[epoch][0] <= t < dur[epoch][1]

    def store_data(self, obs, action, rew):
        if len(self.rew_mat) < self.max_num_samples:
            self.obs_mat.append(obs)
            self.act_mat.append(action)
            self.rew_mat.append(rew)

    def analysis(self):
        """
        performs behavioral analysis relevant for the task
        (i.e. psychometric cuves)
        """
        pass
