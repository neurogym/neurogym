#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:48:19 2019

@author: molano


Perceptual decision-making task, based on

  Bounded integration in parietal cortex underlies decisions even when viewing
  duration is dictated by the environment.
  R Kiani, TD Hanks, & MN Shadlen, JNS 2008.

  http://dx.doi.org/10.1523/JNEUROSCI.4761-07.2008

"""
from __future__ import division

import numpy as np
from gym import spaces
import tasktools
import ngym


class RDM(ngym.ngym):
    def __init__(self, dt=100, timing=[500, 80, 330, 1500, 500], **kwargs):
        super().__init__(dt=dt)
        # Inputs
        self.inputs = tasktools.to_map('FIXATION', 'LEFT', 'RIGHT')

        # Actions
        self.actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT',
                                        'CHOOSE-RIGHT')

        # trial conditions
        self.choices = [-1, 1]
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])

        # Input noise
        self.sigma = np.sqrt(2*100*0.01)

        # Durations
        self.fixation = timing[0]
        self.stimulus_min = timing[1]
        self.stimulus_mean = timing[2]
        self.stimulus_max = timing[3]
        self.decision = timing[4]
        self.mean_trial_duration = self.fixation + self.stimulus_mean +\
            self.decision
        print('mean trial duration: ' + str(self.mean_trial_duration) +
              ' (max num. steps: ' + str(self.mean_trial_duration/self.dt) +
              ')')
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False

        # action and observation spaces
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None

        # set parameters TODO
        # [setattr(self, k, v) for k, v in kwargs]

        # start new trial
        self.trial = self._new_trial()

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        stimulus = tasktools.truncated_exponential(self.rng, self.dt,
                                                   self.stimulus_mean,
                                                   xmin=self.stimulus_min,
                                                   xmax=self.stimulus_max)
        # maximum length of current trial
        self.tmax = self.fixation + stimulus + self.decision
        durations = {
            'fixation': (0, self.fixation),
            'stimulus': (self.fixation, self.fixation + stimulus),
            'decision': (self.fixation + stimulus,
                         self.fixation + stimulus + self.decision),
            }

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        # TODO: allow ground_truth be provided as inputs to _new_trial
        ground_truth = tasktools.choice(self.rng, self.choices)

        coh = tasktools.choice(self.rng, self.cohs)

        return {
            'durations': durations,
            'ground_truth': ground_truth,
            'coh': coh
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
            if (action != self.actions['FIXATE']):
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        # this is an if to allow multiple actions
        if self.in_epoch(self.t, 'decision'):
            # TODO: this part can be simplified
            if action == self.actions['CHOOSE-LEFT']:
                tr_perf = True
                info['new_trial'] = True
                if (trial['ground_truth'] < 0):
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL
            elif action == self.actions['CHOOSE-RIGHT']:
                tr_perf = True
                info['new_trial'] = True
                if (trial['ground_truth'] > 0):
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL
        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        if trial['ground_truth'] < 0:
            high = self.inputs['LEFT']
            low = self.inputs['RIGHT']
        else:
            high = self.inputs['RIGHT']
            low = self.inputs['LEFT']

        obs = np.zeros(len(self.inputs))
        # TODO: maybe allow self.in_epoch(self.t, ['fixation', 'stimulus',...])
        if self.in_epoch(self.t, 'fixation') or\
           self.in_epoch(self.t, 'stimulus'):
            obs[self.inputs['FIXATION']] = 1
        if self.in_epoch(self.t, 'stimulus'):
            obs[high] = self.scale(+trial['coh']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
            obs[low] = self.scale(-trial['coh']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)

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

    # TODO: not used anymore
    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8
