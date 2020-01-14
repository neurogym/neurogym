#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:01:22 2019

@author: molano

GO/NO-GO task based on:

  Active information maintenance in working memory by a sensory cortex
  Xiaoxing Zhang, Wenjun Yan, Wenliang Wang, Hongmei Fan, Ruiqing Hou,
  Yulei Chen, Zhaoqin Chen, Shumin Duan, Albert Compte, Chengyu Li bioRxiv 2018

  https://www.biorxiv.org/content/10.1101/385393v1

"""
from __future__ import division

import numpy as np
from gym import spaces

from neurogym.ops import tasktools
import neurogym as ngym


def get_default_timing():
    return {'fixation': ('constant', 100),
            'stimulus': ('constant', 200),
            'resp_delay': ('constant', 100),
            'decision': ('constant', 100)}


class GNG(ngym.EpochEnv):
    """Go-no-go task."""
    def __init__(self, dt=100, timing=None, **kwargs):
        super().__init__(dt=dt)
        # Actions (fixate, go)
        self.actions = [0, 1]
        # trial conditions
        self.choices = [0, 1]
        # Input noise
        self.sigma = np.sqrt(2*100*0.01)
        # Durations
        default_timing = get_default_timing()
        if timing is not None:
            default_timing.update(timing)
        self.set_epochtiming(default_timing)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_INCORRECT = 0.
        self.R_MISS = 0.
        self.abort = False
        # set action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, ),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        # Trial info
        ground_truth = self.rng.choice(self.choices)

        # Epoch info
        self.add_epoch('fixation', after=0)
        self.add_epoch('stimulus', after='fixation')
        self.add_epoch('resp_delay', after='stimulus')
        self.add_epoch('decision', after='resp_delay', last_epoch=True)

        self.set_ob('fixation', [1, 0, 0])
        tmp =  [1, 0, 0]
        tmp[ground_truth] = 1
        self.set_ob('stimulus', tmp)
        self.set_ob('resp_delay', [1, 0, 0])

        self.set_groundtruth('decision', ground_truth)

        return {
            'ground_truth':  ground_truth,
            }

    def _step(self, action):
        info = {'new_trial': False}
        reward = 0
        obs = self.obs[self.t_ind]
        gt = self.gt[self.t_ind]
        if self.in_epoch('fixation'):
            if action != 0:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action != 0:
                info['new_trial'] = True
                if gt != 0:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_INCORRECT

        return obs, reward, False, info
