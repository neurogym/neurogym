#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-arm Bandit task
TODO: add the actual papers
"""
from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym


class Bandit(ngym.Env):
    def __init__(self, dt=100, n_arm=2, probs=[.9, .1], timing=[]):
        super().__init__(dt=dt)
        # Rewards
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.n_arm = n_arm

        # Reward probabilities
        self.p_high = probs[0]
        self.p_low = probs[1]

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
        rew_high_reward_arm = (self.rng.random() <
                               self.p_high) * self.R_CORRECT
        rew_low_reward_arm = (self.rng.random() < self.p_low) * self.R_CORRECT
        return {
            'rew_high_reward_arm': rew_high_reward_arm,
            'rew_low_reward_arm': rew_low_reward_arm,
            'high_reward_arm': high_reward_arm,
            }

    def _step(self, action):
        trial = self.trial
        info = {'continue': True, 'gt': np.zeros((self.n_arm,))}

        obs = np.zeros(self.observation_space.shape)
        if action == trial['high_reward_arm']:
            reward = trial['rew_high_reward_arm']
        else:
            reward = trial['rew_low_reward_arm']

        # ---------------------------------------------------------------------
        # new trial?
        info['new_trial'] = True
        self.t = 0
        self.num_tr += 1
        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self.trial = self._new_trial()
        return obs, reward, done, info
