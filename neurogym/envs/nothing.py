#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 07:36:17 2020

@author: manuel
"""

from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym


class Nothing(ngym.TrialEnv):
    """Multi-arm bandit task.

    The agent has to select between N actions with different reward
    probabilities.
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41593-018-0147-8',
        'paper_name': 'Prefrontal cortex as a meta-reinforcement learning' +
        ' system',
        'tags': ['n-alternative', 'supervised']
    }

    def __init__(self, dt=100, timing=None, verbose=False):
        super().__init__(dt=dt)
        # Rewards
        self.rewards = {'correct': +1.}
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,),
                                            dtype=np.float32)

        self.verbose = verbose

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.trial = {
            'rew_high_reward_arm': 1,
            'rew_low_reward_arm': 0,
            'high_reward_arm': 0,
            }
        self.trial.update(kwargs)
        self.ob = np.zeros((1, self.observation_space.shape[0]))
        self.gt = np.array([0])
        if self.verbose:
            print('task new trial')

    def _step(self, action):
        trial = self.trial
        info = {'new_trial': True, 'gt': self.gt}

        obs = self.ob[0]
        if action == trial['high_reward_arm']:
            reward = trial['rew_high_reward_arm']
            self.performance = 1
        else:
            reward = trial['rew_low_reward_arm']
        if self.verbose:
            print('task step')
        return obs, reward, False, info
