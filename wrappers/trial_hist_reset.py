#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:08:04 2019

@author: molano
"""

from gym.core import Wrapper
from neurogym.ops import tasktools


class TrHReset(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, rep_prob=(.2, .8)):
        Wrapper.__init__(self, env=env)
        self.env = env
        # we get the original task, in case we are composing wrappers
        env_aux = env
        while env_aux.__class__.__module__.find('wrapper') != -1:
            env_aux = env.env
        self.task = env_aux
        self.rep_prob = rep_prob
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.task.rng, [0, 1])
        self.prev_trial = self.task.trial['ground_truth']

    def _modify_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        trial = self.task.trial
        # rep. probs might depend on previous outcome
        if not self.prev_correct:
            self.curr_block = self.task.rng.randint(0, len(self.rep_prob)-1)
            prev_trial = tasktools.choice(self.task.rng, self.task.choices)
        else:
            prev_trial = self.prev_trial

        if prev_trial == -1:
            probs = (self.rep_prob[self.curr_block],
                     1-self.rep_prob[self.curr_block])
        else:
            probs = (1-self.rep_prob[self.curr_block],
                     self.rep_prob[self.curr_block])

        trial['ground_truth'] = self.task.rng.choices(self.task.choices,
                                                      weights=probs)[0]
        self.prev_trial = trial['ground_truth']

        return trial

    def reset(self):
        return self.task.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['new_trial']:
            info['rep_prob'] = self.rep_prob[self.curr_block]
            self.prev_correct = reward == self.task.R_CORRECT
            self.task.trial = self._modify_trial()
        return obs, reward, done, info

    def seed(self, seed=None):
        self.task.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.task.rng, [0, 1])
