#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:15:02 2019

@author: molano
"""

from gym.core import Wrapper
import numpy as np
main_trans_prob = 0.7
tr_mat = np.zeros((2, 4, 4)) + (1-main_trans_prob)/3
tr_mat[0, 0, 1] = main_trans_prob
tr_mat[0, 1, 2] = main_trans_prob
tr_mat[0, 2, 3] = main_trans_prob
tr_mat[0, 3, 0] = main_trans_prob
tr_mat[1, :, :] = tr_mat[0, :, :].T


class TrialHistory_NAlt(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, trans=tr_mat, block_dur=200,
                 blk_ch_prob=None, pass_blck_info=False):
        Wrapper.__init__(self, env=env)
        self.env = env
        # we get the original task, in case we are composing wrappers
        env_aux = env
        while env_aux.__class__.__module__.find('wrapper') != -1:
            env_aux = env.env
        self.task = env_aux
        self.tr_mat = tr_mat
        # keeps track of the repeating prob of the current block
        self.curr_block = self.task.rng.choice([0, 1])
        # duration of block (in number oif trials)
        self.block_dur = block_dur
        self.prev_trial = self.task.trial['ground_truth']
        self.blk_ch_prob = blk_ch_prob
        self.pass_blck_info = pass_blck_info

    def _modify_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        trial = self.task.trial
        # change rep. prob. every self.block_dur trials
        if self.blk_ch_prob is None:
            if self.task.num_tr % self.block_dur == 0:
                self.curr_block = (self.curr_block + 1) % self.tr_mat.shape[0]
        else:
            if self.task.rng.random() < self.blk_ch_prob:
                self.curr_block = (self.curr_block + 1) % self.tr_mat.shape[0]
        # get probs
        if self.prev_trial == -1:
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
        if self.pass_blck_info:
            obs = np.concatenate((obs, np.array([self.curr_block])))
        return obs, reward, done, info

    def seed(self, seed=None):
        self.task.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = self.task.rng.choice([0, 1])
