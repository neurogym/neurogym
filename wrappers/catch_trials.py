#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:23:36 2019

@author: molano
"""
from gym.core import Wrapper
from neurogym.ops import tasktools
import numpy as np


class CatchTrials(Wrapper):
    """
    introduces catch trials in which the reward for a correct choice
    is modified (by default, is equal to reward for an incorrect choice).
    Note that the wrapper only changes the reward associated to a correct
    answer and does not change the ground truth. Thus, the catch trial would
    not be entirely complete for supervised learning.
    """
    def __init__(self, env, catch_prob=0.01, stim_th=50):
        Wrapper.__init__(self, env=env)
        self.env = env
        # we get the original task, in case we are composing wrappers
        env_aux = env
        while env_aux.__class__.__module__.find('wrapper') != -1:
            env_aux = env.env
        self.task = env_aux
        self.catch_prob = catch_prob
        if stim_th is not None:
            self.stim_th = np.percentile(self.task.cohs, stim_th)
        else:
            self.stim_th = None
        self.R_CORRECT_ORI = self.task.R_CORRECT
        self.catch_trial = False

    def _modify_trial(self):
        trial = self.task.trial
        self.task.R_CORRECT = self.R_CORRECT_ORI
        if self.stim_th is not None:
            if trial['coh'] < self.stim_th:
                self.catch_trial = self.task.rng.random() < self.catch_prob
            else:
                self.catch_trial = False
        else:
            self.catch_trial = self.task.rng.random() < self.catch_prob
        if self.catch_trial:
            self.task.R_CORRECT = self.task.R_FAIL

        return trial

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['new_trial']:
            info['catch_trial'] = self.catch_trial
            _ = self._modify_trial()

        return obs, reward, done, info

    def seed(self, seed=None):
        self.task.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.task.rng, [0, 1])
