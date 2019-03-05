#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:49:35 2019

@author: molano
"""

from gym.core import Wrapper
import tasktools


class compose(Wrapper):
    """
    combines two different tasks
    """
    def __init__(self, env1, env2):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.rep_prob = rep_prob
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.env.rng, [0, 1])
        # duration of block (in number oif trials)
        self.block_dur = block_dur
        self.prev_trial = self.env.trial['ground_truth']

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        trial = self.env._new_trial()
        # change rep. prob. every self.block_dur trials
        if self.env.num_tr % self.block_dur == 0:
            self.curr_block = int(not self.curr_block)

        if self.prev_trial == -1:
            probs = (self.rep_prob[self.curr_block],
                     1-self.rep_prob[self.curr_block])
        else:
            probs = (1-self.rep_prob[self.curr_block],
                     self.rep_prob[self.curr_block])

        trial['ground_truth'] = self.env.rng.choice(self.env.choices,
                                                    p=probs)
        self.prev_trial = trial['ground_truth']

        return trial

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info, new_trial = self.env._step(action)

        if new_trial:
            self.env.trial = self._new_trial()

        return obs, reward, done, info