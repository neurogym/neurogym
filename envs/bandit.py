#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-arm Bandit task
TODO: add the actual papers
"""
from __future__ import division

import numpy as np
from gym import spaces
import tasktools
import ngym


class Bandit(ngym.ngym):
    def __init__(self, dt=100, n_arm=2):
        super().__init__(dt=dt)
        # Rewards
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.n_arm = n_arm

        # Reward probabilities
        self.p_high = 0.9
        self.p_low = 0.1

        self.action_space = spaces.Discrete(n_arm)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.trial = self._new_trial()

    def _new_trial(self, high_reward_arm=0):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        rew_high_reward_arm = (self.rng.rand() < self.p_high) * self.R_CORRECT
        rew_low_reward_arm = (self.rng.rand() < self.p_low) * self.R_CORRECT
        return {
            'rew_high_reward_arm': rew_high_reward_arm,
            'rew_low_reward_arm': rew_low_reward_arm,
            'high_reward_arm': high_reward_arm,
            }

    def _step(self, action):
        trial = self.trial
        info = {'continue': True}
        tr_perf = True

        obs = np.zeros(self.observation_space.shape)
        if action == trial['high_reward_arm']:
            reward = trial['rew_high_reward_arm']
        else:
            reward = trial['rew_low_reward_arm']

        # ---------------------------------------------------------------------
        # new trial?
        new_trial = True
        info['new_trial'] = True
        self.t = 0
        self.num_tr += 1
        # compute perf
        self.perf, self.num_tr_perf =\
            tasktools.compute_perf(self.perf, reward,
                                   self.num_tr_perf, tr_perf)

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info, new_trial

    def step(self, action):
        obs, reward, done, info, new_trial = self._step(action)
        if new_trial:
            self.trial = self._new_trial()
        return obs, reward, done, info
