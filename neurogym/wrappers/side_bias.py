#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:41:50 2019

@author: molano
"""

import numpy as np
from gym.core import Wrapper
import neurogym as ngym
from neurogym.ops import tasktools


# TODO: Need proper testing
class SideBias(ngym.TrialWrapper):
    """Changing the probability of ground truth"""
    def __init__(self, env, prob, block_dur=200):
        """
        Args:
            prob: numpy array (n_block, n_choices)
                within each block, the probability should sum up to 1
            block_dur: number of trials per block
        """
        super().__init__(env)
        try:
            self.choices = self.task.choices
        except AttributeError:
            raise AttributeError('SideBias requires task to have attribute choices')
        assert isinstance(self.task, ngym.TrialEnv), 'Task has to be a TrialEnv'

        self.choice_prob = np.array(prob)
        assert self.choice_prob.shape[1] == len(self.choices),\
            'choice_prob must have shape (n_block, n_choice)'

        self.n_block = self.choice_prob.shape[0]
        self.curr_block = self.task.rng.choice(range(self.n_block))
        self.block_dur = block_dur

    def new_trial(self, **kwargs):
        # change rep. prob. every self.block_dur trials
        if self.task.num_tr % self.block_dur == 0:
            curr_block = self.curr_block
            while curr_block == self.curr_block:
                curr_block = self.task.rng.choice(range(self.n_block))
            self.curr_block = curr_block
        probs = self.choice_prob[self.curr_block]

        kwargs = dict()
        kwargs['ground_truth'] = self.task.rng.choices(self.choices, weights=probs)[0]
        return self.env.new_trial(**kwargs)


class SideBiasObsolete(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, prob=(.2, .8), block_dur=200):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.prob = prob
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.env.rng, [0, 1])
        # duration of block (in number oif trials)
        self.block_dur = block_dur
        self.prev_trial = self.env.trial['ground_truth']

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        trial = self.env._new_trial()
        # change rep. prob. every self.block_dur trials
        if self.env.num_tr % self.block_dur == 0:
            self.curr_block = int(not self.curr_block)

        probs = (1-self.prob[self.curr_block],
                 self.prob[self.curr_block])

        trial['ground_truth'] = self.env.rng.choices(self.env.choices,
                                                     weights=probs)[0]
        return trial

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env._step(action)

        if info['new_trial']:
            info['prob'] = self.prob[self.curr_block]
            self.env.trial = self._new_trial()

        return obs, reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.env.rng, [0, 1])
