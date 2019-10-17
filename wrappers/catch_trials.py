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
    is modified (by default, is equal to reward for an incorrect choice)
    """
    def __init__(self, env, catch_prob=0.01, stim_th=50, rew=None):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.catch_prob = catch_prob
        if stim_th is not None:
            self.stim_th = np.percentile(env.cohs, stim_th)
        else:
            self.stim_th = None
        self.rew = rew
        self.R_CORRECT_ORI = env.R_CORRECT

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        trial = self.env._new_trial()
        self.env.R_CORRECT = self.R_CORRECT_ORI
        if self.stim_th is not None:
            if trial['coh'] <= self.stim_th:
                self.catch_trial = self.env.rng.random() < self.catch_prob
        else:
            self.catch_trial = self.env.rng.random() < self.catch_prob

        return trial

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if self.catch_trial:
            if self.env.in_epoch(self.env.t, 'decision'):
                gt_sign = np.sign(self.env.trial['ground_truth'])
                action_sign = np.sign(self.env.actions[action])
                if gt_sign == action_sign:
                    action = self.env.actions[action]

        obs, reward, done, info = self.env._step(action)

        if info['new_trial']:
            info['catch_trial'] = self.catch_trial
            self.env.trial = self._new_trial()

        return obs, reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.env.rng, [0, 1])


