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
        self.counter = []  # TIP: could this be just a scalar?

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
            self.task.delays
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
        # TIP: you don't need to return this the task is modified and
        # can be access via self.task
        # return self.task

    def counter(self, action):
        # analyzes the last three answers during stage 0
        self.alternate = False
        if self.curr_ph == 0:
            self.counter.append(self.task.actions[action])
            if len(self.counter) > 3:
                del self.counter[0]
            if abs(sum(self.counter)) == 3:
                self.alternate = True
        # return self.alternate

    def _alternate(self, action):
        action_sign = np.sign(self.actions[action])
        self.task.R_CORRECT = self.task.R_FAIL =\
            abs(self.counter[0]+action_sign-action_sign*2)/2
        # return self.task

    def reset(self):
        return self.task.reset()

    def step(self, action):
        # TIP: you might want to put this inside the if below
        #        if self.alternate:
        #            self._alternate(action)
        obs, reward, done, info = self.env._step(action)
        if info['new_trial']:
            # TIP: the following 2 lines are only executed in phase 0, right?
            self.counter(action)  # TIP: this function takes an action
            self._alternate(action)
            self._set_trial_params()
            self.task.trial = self.task._new_trial()
        return obs, reward, done, info
