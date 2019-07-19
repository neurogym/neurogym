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
from neurogym.envs import ngym


class GNG(ngym.ngym):
    def __init__(self, dt=100, timing=(100, 200, 200, 200, 100, 100),
                 **kwargs):
        super().__init__(dt=dt)
        # Actions (fixate, go)
        self.actions = [-1, 1]
        # trial conditions
        self.choices = [-1, 1]
        # Input noise
        self.sigma = np.sqrt(2*100*0.01)
        # Durations
        self.fixation = timing[0]
        self.stimulus_min = timing[1]
        self.stimulus_mean = timing[2]
        self.stimulus_max = timing[3]
        self.resp_delay = timing[4]
        self.decision = timing[5]
        self.mean_trial_duration = self.fixation + self.stimulus_mean +\
            self.resp_delay + self.decision
        if self.fixation == 0 or self.decision == 0 or self.stimulus_mean == 0:
            print('XXXXXXXXXXXXXXXXXXXXXX')
            print('the duration of the fixation, stimulus and decision ' +
                  'periods must be larger than 0')
            print('XXXXXXXXXXXXXXXXXXXXXX')
        print('mean trial duration: ' + str(self.mean_trial_duration) +
              ' (max num. steps: ' +
              str(self.mean_trial_duration/self.dt) + ')')
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_INCORRECT = 0.
        self.R_MISS = 0.
        self.abort = False
        # set action and observation spaces
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, ),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None
        self.trial = self._new_trial()

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        stimulus = tasktools.truncated_exponential(self.rng, self.dt,
                                                   self.stimulus_mean,
                                                   xmin=self.stimulus_min,
                                                   xmax=self.stimulus_max)
        # maximum duration of current trial
        self.tmax = self.fixation + stimulus + self.resp_delay + self.decision
        durations = {
            'fixation':  (0, self.fixation),
            'stimulus':  (self.fixation, self.fixation + stimulus),
            'resp_delay':  (self.fixation + stimulus,
                            self.fixation + stimulus + self.resp_delay),
            'decision':  (self.fixation + stimulus + self.resp_delay,
                          self.tmax),
            }

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        ground_truth = tasktools.choice(self.rng, self.choices)

        return {
            'durations':   durations,
            'ground_truth':  ground_truth,
            }

    # Input scaling
    def scale(self, coh):
        return (1 + coh/100)/2

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'new_trial': False}
        info['gt'] = np.zeros((2,))
        reward = 0
        obs = np.zeros((3,))
        if self.in_epoch(self.t, 'fixation'):
            info['gt'][0] = 1
            obs[0] = 1  # fixation cue only during fixation period
            if self.actions[action] != -1:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch(self.t, 'decision'):
            info['gt'][int((trial['ground_truth']/2+.5))] = 1
            gt_sign = np.sign(trial['ground_truth'])
            action_sign = np.sign(self.actions[action])
            if (action_sign > 0):
                info['new_trial'] = True
                if (gt_sign > 0):
                    reward = self.R_CORRECT
                else:
                    reward = self.R_INCORRECT
        else:
            info['gt'][0] = 1

        if self.in_epoch(self.t, 'stimulus'):
            # observation
            stim = (trial['ground_truth'] > 0) + 1
            obs[stim] = 1

        # ---------------------------------------------------------------------
        # new trial?
        reward, info['new_trial'] = tasktools.new_trial(self.t, self.tmax,
                                                        self.dt,
                                                        info['new_trial'],
                                                        self.R_MISS, reward)
        if info['new_trial']:
            self.t = 0
            self.num_tr += 1
        else:
            self.t += self.dt

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self.trial = self._new_trial()
        return obs, reward, done, info
