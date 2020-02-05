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

import neurogym as ngym


class GoNogo(ngym.PeriodEnv):
    # TODO: Find the original go-no-go paper
    metadata = {
        'description': 'Go/No-Go task in which the subject has either Go' +
        ' (e.g. lick) or not Go depending on which one of two stimuli is' +
        ' presented with.',
        'paper_link': 'https://elifesciences.org/articles/43191',
        'paper_name': 'Active information maintenance in working memory' +
        ' by a sensory cortex',
        'timing': {
            'fixation': ('constant', 0),
            'stimulus': ('constant', 500),
            'resp_delay': ('constant', 500),
            'decision': ('constant', 500)},
        'tags': ['delayed response', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=100, timing=None):
        """
        Go/No-Go task in which the subject has either Go (e.g. lick)
        or not Go depending on which one of two stimuli is presented with.
        dt: Timestep duration. (def: 100 (ms), int)
        timing: Description and duration of periods forming a trial.
        """
        super().__init__(dt=dt, timing=timing)
        # Actions (fixate, go)
        self.actions = [0, 1]
        # trial conditions
        self.choices = [0, 1]

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = -0.5
        self.R_MISS = -0.5  # punishment for miss when trial is GO
        self.abort = False
        # set action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, ),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        # Trial info
        self.trial = {
            'ground_truth': self.rng.choice(self.choices)
        }
        self.trial.update(kwargs)

        # Period info
        self.add_period('fixation', after=0)
        self.add_period('stimulus', after='fixation')
        self.add_period('resp_delay', after='stimulus')
        self.add_period('decision', after='resp_delay', last_period=True)
        # set observations
        self.set_ob('fixation', [1, 0, 0])
        self.set_ob('stimulus', [1, 0, 0])
        self.set_ob('resp_delay', [1, 0, 0])
        ob = self.view_ob('stimulus')
        ob[:, self.trial['ground_truth']+1] = 1
        # if trial is GO the reward is set to R_MISS and  to 0 otherwise
        self.r_tmax = self.R_MISS*self.trial['ground_truth']
        # set ground truth during decision period
        self.set_groundtruth('decision', self.trial['ground_truth'])

    def _step(self, action):
        new_trial = False
        reward = 0
        obs = self.obs_now
        gt = self.gt_now
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if gt != 0:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}
