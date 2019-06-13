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
from neurogym.ops import tasktools
from neurogym.envs import ngym
from gym import spaces


class DPA(ngym.ngym):
    def __init__(self, dt=100,
                 timing=[100, 500, 500, 500, 500, 200, 500]):
        # call ngm __init__ function
        super().__init__(dt=dt)
        # Actions
        self.actions = [-1, 1]
        # trial conditions
        self.dpa_pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]
        # Input noise
        self.sigma = np.sqrt(2*100*0.001)
        # Epoch durations
        self.fixation = timing[0]
        self.dpa1 = timing[1]
        self.delay_min = timing[2]
        self.delay_max = timing[3]
        stimulus_mean = (timing[2]+timing[3])/2
        self.dpa2 = timing[4]
        self.resp_delay = timing[5]
        self.decision = timing[6]
        self.delay_mean = (self.delay_min + self.delay_max)/2
        self.mean_trial_duration = self.fixation + self.dpa1 +\
            self.delay_mean + self.dpa2 + self.resp_delay + self.decision
        if self.fixation == 0 or self.decision == 0 or stimulus_mean == 0:
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
        self.R_INCORRECT = -1.
        self.R_MISS = 0.
        self.abort = False
        # set action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-1., 1, shape=(5, ),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None

        # start new trial
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
            ground_truth = 1
        else:
            ground_truth = -1

        return {
            'durations': durations,
            'ground_truth':     ground_truth,
            'pair':     pair
            }

    def _step(self, action):
        trial = self.trial

        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        # epochs = trial['epochs']
        info = {'new_trial': False}
        reward = 0
        obs = np.zeros((5,))
        if self.in_epoch(self.t, 'fixation'):
            obs[0] = 1
            if self.actions[action] != -1:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        if self.in_epoch(self.t, 'decision'):
            gt_sign = np.sign(trial['ground_truth'])
            action_sign = np.sign(self.actions[action])
            if (gt_sign > 0) and (action_sign > 0):
                reward = self.R_CORRECT
                info['new_trial'] = True
        else:
            obs[0] = 1

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------
        # if self.t in epochs['dpa1']:
        if self.in_epoch(self.t, 'dpa1'):
            dpa1, _ = trial['pair']
            obs[dpa1] = 1
        # if self.t in epochs['dpa2']:
        if self.in_epoch(self.t, 'dpa2'):
            _, dpa2 = trial['pair']
            obs[dpa2] = 1
        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['new_trial'],
                                                self.R_MISS, reward)
        info['gt'] = np.zeros((2,))
        if new_trial:
            info['new_trial'] = True
            info['gt'][int((trial['ground_truth']/2+.5))] = 1
            self.t = 0
            self.num_tr += 1
        else:
            info['gt'][0] = 1
            self.t += self.dt
        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self.trial = self._new_trial()
        return obs, reward, done, info
