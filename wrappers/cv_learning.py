#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:45:33 2019

@author: molano
"""

import numpy as np
from gym.core import Wrapper


class CurriculumLearning(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, perf_w=10):
        Wrapper.__init__(self, env=env)
        self.env = env
        # we get the original task, in case we are composing wrappers
        env_aux = env
        while env_aux.__class__.__module__.find('wrapper') != -1:
            env_aux = env.env
        self.task = env_aux
        self.curr_ph = 0
        self.curr_perf = 0
        self.perf_window = perf_w
        self.counter = 0

    def _set_trial_params(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        if self.curr_ph == 0:
            # no stim, reward is in both left and right
            # agent cannot go N times in a row to the same side
            self.task.stimulus_mean = 0
            self.task.stimulus_min = 0
            self.task.stimulus_max = 0
            # self.task.delays
            self.task.R_FAIL = self.task.R_CORRECT
        elif self.curr_ph == 1:
            # there is stim but first answer is not penalized
            self.task.stimulus_min = self.task.timing[1]
            self.task.stimulus_mean = self.task.timing[2]
            self.task.stimulus_max = self.task.timing[3]
            self.task.R_FAIL = 0
            self.task.firstcounts = False
            self.task.cohs = np.array([100])
        elif self.curr_ph == 2:
            # first answer counts
            self.task.R_FAIL = -self.task.R_CORRECT
            self.task.firstcounts = True
        elif self.curr_ph == 3:
            self.task.delays = [1000, 5000, 10000]
        elif self.curr_ph == 4:
            self.task.coh = np.array([0, 6.4, 12.8, 25.6, 51.2]) *\
                self.task.stimEv

    def count(self, action):
        # analyzes the last three answers during stage 0
        # self.alternate = False
        new = self.task.actions[action]
        if np.sign(self.counter) == np.sign(new):
            self.counter += new
        else:
            self.counter = new

    def reset(self):
        return self.task.reset()

    def step(self, action):
        obs, reward, done, info = self.env._step(action)
        if info['new_trial']:
            if self.curr_ph == 0:
                self.count(action)
                if np.abs(self.counter) >= self.max_num_reps:
                    reward = self.task.FAIL
            self._set_trial_params()
            self.task.trial = self.task._new_trial()
        return obs, reward, done, info
