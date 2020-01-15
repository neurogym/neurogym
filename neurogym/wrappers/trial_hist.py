#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:48:59 2019

@author: molano
"""

from gym.core import Wrapper
import numpy as np

import neurogym as ngym


# TODO: Remain to be finished
class TrialHistory(ngym.TrialWrapper):
    """Change ground truth probability based on outcome."""
    def __init__(self, env, rep_prob=(.2, .8), block_dur=200,
                 blk_ch_prob=None, ae_probs=None, pass_blck=False):
        super().__init__(env)
        if ae_probs is None:
            ae_probs = rep_prob

        self.rep_prob = rep_prob
        self.ae_probs = ae_probs
        # keeps track of the repeating prob of the current block
        self.curr_block = self.task.rng.choice([0, 1])
        # duration of block (in number oif trials)
        self.block_dur = block_dur
        self.prev_trial = self.task.trial['ground_truth']
        self.blk_ch_prob = blk_ch_prob
        self.pass_blck_info = pass_blck

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        info = kwargs['info']
        # change rep. prob. every self.block_dur trials
        if self.blk_ch_prob is None:
            if self.task.num_tr % self.block_dur == 0:
                self.curr_block = (self.curr_block + 1) % len(self.rep_prob)
        else:
            if self.task.rng.random() < self.blk_ch_prob:
                self.curr_block = (self.curr_block + 1) % len(self.rep_prob)

        # rep. probs might depend on previous outcome
        if info['correct']:
            if self.prev_trial == -1:
                probs = (self.rep_prob[self.curr_block],
                         1-self.rep_prob[self.curr_block])
            else:
                probs = (1-self.rep_prob[self.curr_block],
                         self.rep_prob[self.curr_block])
        else:
            if self.prev_trial == -1:
                probs = (self.ae_probs[self.curr_block],
                         1-self.ae_probs[self.curr_block])
            else:
                probs = (1-self.ae_probs[self.curr_block],
                         self.ae_probs[self.curr_block])

        ground_truth = self.task.rng.choices(self.task.choices,
                                             weights=probs)[0]
        self.prev_trial = ground_truth
        kwargs.update({'groundtruth': ground_truth})
        self.env.new_trial(**kwargs)


class TrialHistoryObsolete(Wrapper):
    """
    modifies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, rep_prob=(.2, .8), block_dur=200,
                 blk_ch_prob=None, ae_probs=None, pass_blck=False):
        if ae_probs is None:
            ae_probs = rep_prob
        Wrapper.__init__(self, env=env)
        self.env = env
        # we get the original task, in case we are composing wrappers
        env_aux = env
        while env_aux.__class__.__module__.find('wrapper') != -1:
            env_aux = env.env
        self.task = env_aux
        self.rep_prob = rep_prob
        self.ae_probs = ae_probs
        # keeps track of the repeating prob of the current block
        self.curr_block = self.task.rng.choice([0, 1])
        # duration of block (in number oif trials)
        self.block_dur = block_dur
        self.prev_trial = self.task.trial['ground_truth']
        self.blk_ch_prob = blk_ch_prob
        self.pass_blck_info = pass_blck

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        # change rep. prob. every self.block_dur trials
        if self.blk_ch_prob is None:
            if self.task.num_tr % self.block_dur == 0:
                self.curr_block = (self.curr_block + 1) % len(self.rep_prob)
        else:
            if self.task.rng.random() < self.blk_ch_prob:
                self.curr_block = (self.curr_block + 1) % len(self.rep_prob)

        # rep. probs might depend on previous outcome
        if self.prev_correct:
            if self.prev_trial == -1:
                probs = (self.rep_prob[self.curr_block],
                         1-self.rep_prob[self.curr_block])
            else:
                probs = (1-self.rep_prob[self.curr_block],
                         self.rep_prob[self.curr_block])
        else:
            if self.prev_trial == -1:
                probs = (self.ae_probs[self.curr_block],
                         1-self.ae_probs[self.curr_block])
            else:
                probs = (1-self.ae_probs[self.curr_block],
                         self.ae_probs[self.curr_block])

        ground_truth = self.task.rng.choices(self.task.choices,
                                             weights=probs)[0]
        self.prev_trial = ground_truth
        kwargs.update({'gt': ground_truth})
        self.env.new_trial(**kwargs)

    def reset(self):
        return self.task.reset()

    def _step(self, action):
        return self.env._step(action)

    def step(self, action):
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            info['rep_prob'] = self.rep_prob[self.curr_block]
            self.prev_correct = reward == self.task.R_CORRECT
            self.new_trial()
        if self.pass_blck_info:
            obs = np.concatenate((obs, np.array([self.curr_block])))
        return obs, reward, done, info

    def seed(self, seed=None):
        self.task.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = self.task.rng.choice([0, 1])
