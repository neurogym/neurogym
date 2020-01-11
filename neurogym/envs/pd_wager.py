#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:31:14 2019

@author: molano

Perceptual decision-making with postdecision wagering, based on

  Representation of confidence associated with a decision by
  neurons in the parietal cortex.
  R. Kiani & M. N. Shadlen, Science 2009.

  http://dx.doi.org/10.1126/science.1169405

"""
from __future__ import division

import numpy as np
from gym import spaces

from neurogym.ops import tasktools
import neurogym as ngym


class PDWager(ngym.EpochEnv):
    def __init__(self, dt=100, timing=(750, 100, 180, 800, 1200, 1350,
                                       1800, 500, 575, 750, 500)):
        # call ngm __init__ function
        super().__init__(dt=dt)
#        # Actions
#        self.actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT',
#                                        'CHOOSE-RIGHT', 'CHOOSE-SURE')
        # Actions (fixate, left, right, sure)
        self.actions = [0, -1, 1, 2]
        # trial conditions
        self.wagers = [True, False]
        self.choices = [-1, 1]
        self.cohs = [0, 3.2, 6.4, 12.8, 25.6, 51.2]

        # Input noise
        self.sigma = np.sqrt(2*100*0.01)

        # TODO: this is terrible looking, need change
        # Durations
        self.fixation = timing[0]  # 750
        self.stimulus_min = timing[1]  # 100
        self.stimulus_mean = timing[2]  # 180
        self.stimulus_max = timing[3]  # 800
        self.delay_min = timing[4]  # 1200
        self.delay_mean = timing[5]  # 1350
        self.delay_max = timing[6]  # 1800
        self.sure_min = timing[7]  # 500
        self.sure_mean = timing[8]  # 575
        self.sure_max = timing[9]  # 750
        self.decision = timing[10]  # 500
        self.mean_trial_duration = self.fixation + self.stimulus_mean +\
            self.delay_mean + self.decision

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_MISS = 0.
        self.R_FAIL = 0.
        self.abort = False
        self.R_SURE = 0.7*self.R_CORRECT

        # set action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4, ),
                                            dtype=np.float32)

    def __str__(self):
        string = ''
        if self.fixation == 0 or self.decision == 0 or self.stimulus_mean == 0:
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
            string += 'the duration of all periods must be larger than 0\n'
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        string += 'mean trial duration: ' + str(self.mean_trial_duration) + '\n'
        string += 'max num. steps: ' + str(self.mean_trial_duration/self.dt)
        return string

    # Input scaling
    def scale(self, coh):
        return (1 + coh/100)/2

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Wager or no wager?
        # ---------------------------------------------------------------------
        wager = self.rng.choice(self.wagers)
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        stimulus = self.stimulus_min +\
            tasktools.trunc_exp(self.rng, self.dt,
                                            self.stimulus_mean,
                                            xmax=self.stimulus_max)

        delay = tasktools.trunc_exp(self.rng, self.dt,
                                                self.delay_mean,
                                                xmin=self.delay_min,
                                                xmax=self.delay_max)

        self.add_epoch('fixation', self.fixation, start=0)
        self.add_epoch('stimulus', stimulus, after='fixation')
        self.add_epoch('delay', delay, after='stimulus')
        self.add_epoch('decision', self.decision, after='delay', last_epoch=True)

        if wager:
            sure_onset = \
                tasktools.trunc_exp(self.rng, self.dt,
                                    self.sure_mean,
                                    xmin=self.sure_min,
                                    xmax=self.sure_max)
            self.add_epoch('pre_sure', duration=sure_onset, after='stimulus')
            self.add_epoch('sure', duration=10000, after='pre_sure')

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        ground_truth = self.rng.choice(self.choices)

        coh = self.rng.choice(self.cohs)

        return {
            'wager':      wager,
            'ground_truth': ground_truth,
            'coh':        coh
            }

    def _step(self, action):
        trial = self.trial
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'new_trial': False}
        # ground truth signal is not well defined in this task
        info['gt'] = np.zeros((4,))
        # rewards
        reward = 0
        # observations
        obs = np.zeros((4,))
        if self.in_epoch('fixation'):
            obs[0] = 1
            if self.actions[action] != 0:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if self.actions[action] == 2:
                if trial['wager']:
                    reward = self.R_SURE
                else:
                    reward = self.R_ABORTED
            else:
                gt_sign = np.sign(trial['ground_truth'])
                action_sign = np.sign(self.actions[action])
                if gt_sign == action_sign:
                    reward = self.R_CORRECT
                elif gt_sign == -action_sign:
                    reward = self.R_FAIL
            info['new_trial'] = self.actions[action] != 0

        if self.in_epoch('delay'):
            obs[0] = 1
        elif self.in_epoch('stimulus'):
            high = (trial['ground_truth'] > 0) + 1
            low = (trial['ground_truth'] < 0) + 1
            obs[high] = self.scale(+trial['coh']) +\
                self.rng.gauss(mu=0, sigma=self.sigma)/np.sqrt(self.dt)
            obs[low] = self.scale(-trial['coh']) +\
                self.rng.gauss(mu=0, sigma=self.sigma)/np.sqrt(self.dt)
        if trial['wager'] and self.in_epoch('sure'):
            obs[3] = 1

        return obs, reward, False, info
