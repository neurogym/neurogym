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
    def __init__(self, env, catch_prob=0.01, stim_th=50, start=0):
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
        # number of trials after which the prob. of catch trials is != 0
        self.start = start

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
            if self.task.num_tr > self.start:
                info['catch_trial'] = self.catch_trial
                _ = self._modify_trial()
            else:
                info['catch_trial'] = False
        return obs, reward, done, info

    def seed(self, seed=None):
        self.task.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = self.task.rng.choice([0, 1])


if __name__ == '__main__':
    data = np.load('/home/molano/neurogym/results/example1/' +
                   'PassReward0_bhvr_data_1000.npz')
    stim = data['stimulus'][:, 1:3]
    stim = np.diff(stim, axis=1)
    inds = np.where(data['catch_trial'] == 1)[0]
    print(np.mean(np.abs(stim)))
    print(np.mean(np.abs(stim[inds])))

    print(data['reward'][inds])
    #    ut.get_fig(display_mode=1)
    #    gt = np.argmax(data['correct_side'], axis=1)
    #    reps = an.get_repetitions(gt)
    #    plt.plot(reps)
    #    plt.plot(np.convolve(reps, np.ones(100,)/100,  mode='same'))
    
