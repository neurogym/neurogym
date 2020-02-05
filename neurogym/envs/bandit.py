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


class Bandit(ngym.TrialEnv):
    metadata = {
        'description': 'The agent has to select between N actions' +
        ' with different reward probabilities.',
        'paper_link': 'https://www.nature.com/articles/s41593-018-0147-8',
        'paper_name': 'Prefrontal cortex as a meta-reinforcement learning' +
        ' system',
        'tags': ['n-alternative', 'supervised']
    }

    def __init__(self, dt=100, n_arm=2, probs=(.9, .1), gt_arm=0,
                 timing=None):
        """
        The agent has to select between N actions with different reward
        probabilities.
        dt: Timestep duration. (def: 100 (ms), int)
        n_arms: Number of arms. (def: 2, int)
        probs: Reward probabilities for each arm. (def: (.9, .1), tuple)
        gt_arm: High reward arm. (def: 0, int)
        timing: Description and duration of periods forming a trial.
        """
        super().__init__(dt=dt)
        if timing is not None:
            print('Warning: Bandit task does not require timing variable.')
        # Rewards
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.n_arm = n_arm
        self.gt_arm = gt_arm
        # Reward probabilities
        self.p_high = probs[0]
        self.p_low = probs[1]

        self.action_space = spaces.Discrete(n_arm)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        rew_high_reward_arm = (self.rng.random() <
                               self.p_high) * self.R_CORRECT
        rew_low_reward_arm = (self.rng.random() < self.p_low) * self.R_CORRECT
        self.trial = {
            'rew_high_reward_arm': rew_high_reward_arm,
            'rew_low_reward_arm': rew_low_reward_arm,
            'high_reward_arm': self.gt_arm,
            }
        self.trial.update(kwargs)
        self.obs = np.zeros(self.observation_space.shape)

    def _step(self, action):
        trial = self.trial
        info = {'continue': True, 'gt': np.zeros((self.n_arm,))}

        obs = self.obs
        if action == trial['high_reward_arm']:
            reward = trial['rew_high_reward_arm']
        else:
            reward = trial['rew_low_reward_arm']

        # new trial?
        info['new_trial'] = True
        info['gt'][self.gt_arm] = 1
        return obs, reward, False, info
