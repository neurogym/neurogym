#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:25:08 2019

@author: molano


Delay Pair Association (DPA) task based on:

  Active information maintenance in working memory by a sensory cortex
  Xiaoxing Zhang, Wenjun Yan, Wenliang Wang, Hongmei Fan, Ruiqing Hou,
  Yulei Chen, Zhaoqin Chen, Shumin Duan, Albert Compte, Chengyu Li bioRxiv 2018

  https://www.biorxiv.org/content/10.1101/385393v1

"""
import numpy as np
import tasktools
import ngym
from gym import spaces


class DPA(ngym.ngym):
    def __init__(self, dt=100):
        # call ngm __init__ function
        super().__init__(dt=dt)
        # Inputs
        self.inputs = tasktools.to_map('FIXATION', 'S1', 'S2', 'S3', 'S4')
        # Actions
        self.actions = tasktools.to_map('NO_GO', 'GO')
        # trial conditions
        self.dpa_pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]
        # Input noise
        self.sigma = np.sqrt(2*100*0.001)
        # Epoch durations
        self.fixation = 0
        self.dpa1 = 500
        self.delay_min = 500  # Original paper: 13000
        self.delay_max = 500
        self.delay_mean = (self.delay_min + self.delay_max) / 2
        self.dpa2 = 500
        self.resp_delay = 500
        self.decision = 500
        self.mean_trial_duration = self.fixation + self.dpa1 +\
            self.delay_mean + self.dpa2 + self.resp_delay + self.decision
        print('mean trial duration: ' + str(self.mean_trial_duration) +
              ' (max num. steps: ' + str(self.mean_trial_duration/self.dt) +
              ')')

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_INCORRECT = -1.
        self.R_MISS = 0.
        self.abort = False

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-1., 1, shape=(5, ),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None

        self.trial = self._new_trial()

    def _new_trial(self):
        # -------------------------------------------------------------------------
        # Epochs
        # --------------------------------------------------------------------------

        delay = tasktools.uniform(self.rng, self.dt, self.delay_min,
                                  self.delay_max)
        # maximum duration of current trial
        self.tmax = self.fixation + self.dpa1 + delay + self.dpa2 +\
            self.resp_delay + self.decision

        durations = {
            'fixation':   (0, self.fixation),
            'dpa1':         (self.fixation, self.fixation + self.dpa1),
            'delay':      (self.fixation + self.dpa1,
                           self.fixation + self.dpa1 + delay),
            'dpa2':         (self.fixation + self.dpa1 + delay,
                             self.fixation + self.dpa1 + delay + self.dpa2),
            'resp_delay': (self.fixation + self.dpa1 + delay + self.dpa2,
                           self.fixation + self.dpa1 + delay + self.dpa2 +
                           self.resp_delay),
            'decision':   (self.fixation + self.dpa1 + delay + self.dpa2 +
                           self.resp_delay, self.tmax),
            }

        pair = tasktools.choice(self.rng, self.dpa_pairs)

        if np.diff(pair)[0] == 2:
            ground_truth = 'GO'
        else:
            ground_truth = 'NO_GO'

        return {
            'durations': durations,
            'ground_truth':     ground_truth,
            'pair':     pair
            }

    def scale(self, f):
        return (f - self.fmin)/(self.fmax - self.fmin)

    def scale_p(self, f):
        return (1 + self.scale(f))/2

    def scale_n(self, f):
        return (1 - self.scale(f))/2

    def _step(self, action):
        trial = self.trial

        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        # epochs = trial['epochs']
        info = {'new_trial': False}
        reward = 0
        tr_perf = False
        # if self.t not in epochs['decision']:
        if self.in_epoch(self.t, 'fixation'):
            if (action != self.actions['NO_GO']):
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        if self.in_epoch(self.t, 'decision'):
            # print('decision period')
            if action == self.actions['GO']:
                tr_perf = True
                info['new_trial'] = True
                if (trial['ground_truth'] == 'GO'):
                    reward = self.R_CORRECT
                else:
                    reward = self.R_INCORRECT

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------
        dpa1, dpa2 = trial['pair']
        obs = np.zeros(len(self.inputs))
        # if self.t not in epochs['decision']:
        if not self.in_epoch(self.t, 'decision'):
            obs[self.inputs['FIXATION']] = 1
        # if self.t in epochs['dpa1']:
        if self.in_epoch(self.t, 'dpa1'):
            # TODO: Do we need self.inputs?
            obs[dpa1] = 1
        # if self.t in epochs['dpa2']:
        if self.in_epoch(self.t, 'dpa2'):
            obs[dpa2] = 1
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

        return p_decision >= 0.99 and p_correct >= 0.97
