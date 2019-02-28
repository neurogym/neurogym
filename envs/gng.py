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
    # Inputs
    inputs = tasktools.to_map('FIXATION', 'S1', 'S2')

    # Actions
    actions = tasktools.to_map('FIXATE', 'NO_GO', 'GO')

    # Trial conditions
    go_nogos = [-1, 1]

    # Input noise
    sigma = np.sqrt(2*100*0.01)

    # Durations
    fixation = 500
    stimulus_min = 500
    stimulus_mean = 501
    stimulus_max = 502
    resp_delay = 500
    decision = 500
    tmax = fixation + stimulus_max + resp_delay + decision

    # Rewards
    R_ABORTED = -1.
    R_CORRECT = +1.
    R_FAIL = 0.
    R_MISS = 0.

    def __init__(self, dt=100):
        super().__init__(dt=dt)
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, ),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.steps_beyond_done = None

        self.trial = self._new_trial(self.rng, self.dt)
        print('------------------------')
        print('Go/No-Go task')
        print('------------------------')

    def _new_trial(self, rng, dt, context={}):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        stimulus = context.get('stimulus')
        if stimulus is None:
            stimulus = tasktools.truncated_exponential(rng, dt,
                                                       self.stimulus_mean,
                                                       xmin=self.stimulus_min,
                                                       xmax=self.stimulus_max)
        durations = {
            'fixation':  (0, self.fixation),
            'stimulus':  (self.fixation, self.fixation + stimulus),
            'resp_delay':  (self.fixation + stimulus,
                            self.fixation + stimulus + self.resp_delay),
            'decision':  (self.fixation + stimulus + self.resp_delay,
                          self.fixation + stimulus + self.resp_delay +
                          self.decision),
            'tmax':      self.tmax
            }
        time, epochs = tasktools.get_epochs_idx(dt, durations)

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        go_nogo = context.get('go_nogo')
        if go_nogo is None:
            go_nogo = rng.choice(self.go_nogos)

        return {
            'durations':   durations,
            'time':        time,
            'epochs':      epochs,
            'go_nogo':  go_nogo,
            }

    # Input scaling
    def scale(self, coh):
        return (1 + coh/100)/2

    def step(self, action):
        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        trial = self.trial
        epochs = trial['epochs']
        status = {'continue': True}
        reward = 0
        tr_perf = False
        if self.t-1 not in epochs['decision']:
            if action != self.actions['FIXATE']:
                status['continue'] = False
                reward = self.R_ABORTED
        elif self.t-1 in epochs['decision']:
            if action == self.actions['GO']:
                tr_perf = True
                status['continue'] = False
                status['choice'] = 'GO'
                status['t_choice'] = self.t-1
                status['correct'] = (trial['go_nogo'] > 0)
                if status['correct']:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL
            elif action == self.actions['NO_GO']:
                tr_perf = True
                status['continue'] = False
                status['choice'] = 'NG'
                status['t_choice'] = self.t-1
                status['correct'] = (trial['go_nogo'] < 0)
                if status['correct']:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL
        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        if trial['go_nogo'] < 0:
            stim = self.inputs['S1']
        else:
            stim = self.inputs['S2']

        obs = np.zeros(len(self.inputs))
        if self.t in epochs['fixation'] or self.t in epochs['stimulus']:
            obs[self.inputs['FIXATION']] = 1
        if self.t in epochs['stimulus']:
            obs[stim] = 1

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial, self.t, self.perf, self.num_tr, self.num_tr_perf =\
            tasktools.new_trial(self.t, self.tmax, self.dt, status['continue'],
                                self.R_MISS, self.num_tr, self.perf, reward,
                                self.p_stp, self.num_tr_perf, tr_perf)

        if new_trial:
            self.trial = self._new_trial(self.rng, self.dt)

        done = False  # TODO: revisit
        return obs, reward, done, status

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8
