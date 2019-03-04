#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:48:59 2019

@author: molano
"""

from gym.core import Wrapper
import tasktools

class TrialHistory(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, rep_prob=(.2, .8), block_dur=200):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.rep_prob = rep_prob
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.rng, [0, 1])
        # duration of block (in number oif trials)
        self.block_dur = block_dur

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        trial = self.env._new_trial()
        if self.prev_trial == -1:
            probs = (self.rep_prob[self.curr_block],
                     1-self.rep_prob[self.curr_block])
        else:
            probs = (1-self.rep_prob[self.curr_block],
                     self.rep_prob[self.curr_block])

        left_right = self.rng.choice(self.left_rights,
                                     p=probs)

        self.prev_trial = trial['ground_truth']

        return trial
