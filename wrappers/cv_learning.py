#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:45:33 2019

@author: molano
"""


from gym.core import Wrapper
from neurogym.ops import tasktools


class CurriculumLearning(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, rep_prob=(.2, .8), block_dur=200,
                 blk_ch_prob=None, ae_probs=None, perf_w=10):
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
            self.task.R_FAIL = self.task.R_CORRECT
        elif self.curr_ph == 1:
            
        elif self.curr_ph == 2:

        else:
            

        return trial

    def reset(self):
        return self.task.reset()

    def step(self, action):
        obs, reward, done, info = self.env._step(action)
        if info['new_trial']:
            self._set_trial_params()
            self.task.trial = self.task._new_trial()
        return obs, reward, done, info

    def seed(self, seed=None):
        self.task.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.task.rng, [0, 1])
