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
import tasktools
import ngym


class GNG(ngym.ngym):
    def __init__(self, dt=100):
        super().__init__(dt=dt)
        # Inputs
        self.inputs = tasktools.to_map('FIXATION', 'S1', 'S2')

        # Actions
        self.actions = tasktools.to_map('NO_GO', 'GO')

        # trial conditions
        self.choices = [-1, 1]

        # Input noise
        self.sigma = np.sqrt(2*100*0.01)

        # Durations
        self.fixation = 0
        self.stimulus_min = 200
        self.stimulus_mean = 200
        self.stimulus_max = 200
        self.resp_delay = 200
        self.decision = 500
        self.mean_trial_duration = self.fixation + self.stimulus_mean +\
            self.resp_delay + self.decision
        print('mean trial duration: ' + str(self.mean_trial_duration) +
              ' (max num. steps: ' + str(self.mean_trial_duration/self.dt) +
              ')')
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_INCORRECT = -1.
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
        # Reward
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'new_trial': False}
        reward = 0
        tr_perf = False
        if self.in_epoch(self.t, 'fixation'):
            if (action != self.actions['NO_GO']):
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        if self.in_epoch(self.t, 'decision'):
            if action == self.actions['GO']:
                tr_perf = True
                info['new_trial'] = True
                if (trial['ground_truth'] > 0):
                    reward = self.R_CORRECT
                else:
                    reward = self.R_INCORRECT

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        if trial['ground_truth'] < 0:
            stim = self.inputs['S1']
        else:
            stim = self.inputs['S2']

        obs = np.zeros(len(self.inputs))
        if self.in_epoch(self.t, 'fixation') or\
           self.in_epoch(self.t, 'stimulus'):
            obs[self.inputs['FIXATION']] = 1
        if self.in_epoch(self.t, 'stimulus'):
            obs[stim] = 1

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['new_trial'],
                                                self.R_MISS, reward)

        if new_trial:
            info['new_trial'] = True
            info['gt'] = trial['ground_truth']
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward,
                                       self.num_tr_perf, tr_perf)
        else:
            self.t += self.dt

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info, new_trial

    def step(self, action):
        obs, reward, done, info, new_trial = self._step(action)
        if new_trial:
            self.trial = self._new_trial()
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8
