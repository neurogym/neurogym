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
import ngym
from gym import spaces
import numpy as np

import tasktools


class PDWager(ngym.ngym):
    def __init__(self, dt=100):
        # call ngm __init__ function
        super().__init__(dt=dt)
        # Inputs
        self.inputs = tasktools.to_map('FIXATION', 'LEFT', 'RIGHT', 'SURE')

        # Actions
        self.actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT',
                                        'CHOOSE-RIGHT', 'CHOOSE-SURE')

        # trial conditions
        self.wagers = [True, False]
        self.choices = [-1, 1]
        self.cohs = [0, 3.2, 6.4, 12.8, 25.6, 51.2]

        # Input noise
        self.sigma = np.sqrt(2*100*0.01)

        # Durations
        self.fixation = 750
        self.stimulus_min = 100
        self.stimulus_mean = 180
        self.stimulus_max = 800
        self.delay_min = 1200
        self.delay_mean = 1350
        self.delay_max = 1800
        self.sure_min = 500
        self.sure_mean = 575
        self.sure_max = 750
        self.decision = 500
        self.mean_trial_duration = self.fixation + self.stimulus_mean +\
            self.delay_mean + self.decision
        print('mean trial duration: ' + str(self.mean_trial_duration) +
              ' (max num. steps: ' + str(self.mean_trial_duration/self.dt) +
              ')')
        # Rewards
        self.R_ABORTED = -1.
        self.R_CORRECT = +1.
        self.R_MISS = 0.
        self.abort = False
        self.R_SURE = 0.7*self.R_CORRECT

        # set action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4, ),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.trial = self._new_trial()

    # Input scaling
    def scale(self, coh):
        return (1 + coh/100)/2

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Wager or no wager?
        # ---------------------------------------------------------------------

        wager = tasktools.choice(self.rng, self.wagers)

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        stimulus = self.stimulus_min +\
            tasktools.truncated_exponential(self.rng, self.dt,
                                            self.stimulus_mean,
                                            xmax=self.stimulus_max)

        delay = tasktools.truncated_exponential(self.rng, self.dt,
                                                self.delay_mean,
                                                xmin=self.delay_min,
                                                xmax=self.delay_max)

        if wager:
            sure_onset =\
                tasktools.truncated_exponential(self.rng, self.dt,
                                                self.sure_mean,
                                                xmin=self.sure_min,
                                                xmax=self.sure_max)
        # maximum duration of current trial
        self.tmax = self.fixation + stimulus + delay + self.decision
        # TODO: are these periods correct?
        durations = {
            'fixation':  (0, self.fixation),
            'stimulus':  (self.fixation, self.fixation + stimulus),
            'delay':     (self.fixation + stimulus,
                          self.fixation + stimulus + delay),
            'decision':  (self.fixation + stimulus + delay,
                          self.tmax),
            }
        if wager:
            durations['sure'] = (self.fixation + stimulus + sure_onset,
                                 self.tmax)

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        ground_truth = tasktools.choice(self.rng, self.choices)

        coh = tasktools.choice(self.rng, self.cohs)

        return {
            'durations':  durations,
            'wager':      wager,
            'ground_truth': ground_truth,
            'coh':        coh
            }

    def _step(self, action):
        trial = self.trial
        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------

        # epochs = trial['epochs']
        info = {'continue': True}
        reward = 0
        tr_perf = False
        if self.in_epoch(self.t, 'fixation'):
            if (action != self.actions['FIXATE']):
                info['continue'] = not self.abort
                reward = self.R_ABORTED
        if self.in_epoch(self.t, 'decision'):
            if action == self.actions['CHOOSE-LEFT']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'L'
                info['t_choice'] = self.t
                info['correct'] = (trial['ground_truth'] < 0)
                if info['correct']:
                    reward = self.R_CORRECT
            elif action == self.actions['CHOOSE-RIGHT']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'R'
                info['t_choice'] = self.t
                info['correct'] = (trial['ground_truth'] > 0)
                if info['correct']:
                    reward = self.R_CORRECT
            elif action == self.actions['CHOOSE-SURE']:
                tr_perf = True
                info['continue'] = False
                if trial['wager']:
                    info['choice'] = 'S'
                    info['t_choice'] = self.t
                    reward = self.R_SURE
                else:
                    reward = self.R_ABORTED

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
        if (self.in_epoch(self.t, 'fixation') or
           self.in_epoch(self.t, 'stimulus') or
                self.in_epoch(self.t, 'delay')):
            obs[self.inputs['FIXATION']] = 1
        if self.in_epoch(self.t, 'stimulus'):
            obs[high] = self.scale(+trial['coh']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
            obs[low] = self.scale(-trial['coh']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
        if trial['wager'] and self.in_epoch(self.t, 'sure'):
            obs[self.inputs['SURE']] = 1

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['continue'],
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
        p_answer = perf.n_answer/perf.n_trials
        p_correct = tasktools.divide(perf.n_correct, perf.n_decision)
        p_sure = tasktools.divide(perf.n_sure, perf.n_sure_decision)

        return p_answer >= 0.99 and p_correct >= 0.79 and 0.4 < p_sure <= 0.5
