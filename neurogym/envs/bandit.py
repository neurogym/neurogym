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
                 rewards=None, timing=None):
        """
        The agent has to select between N actions with different reward
        probabilities.
        """
        super().__init__(dt=dt)
        if timing is not None:
            print('Warning: Bandit task does not require timing variable.')

        # Rewards
        self.rewards = {'correct': +1.}
        if rewards:
            self.rewards.update(rewards)
        self.rewards['correct'] = self.rewards['correct']

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
                               self.p_high) * self.rewards['correct']
        rew_low_reward_arm = (self.rng.random() < self.p_low) * self.rewards['correct']
        self.trial = {
            'rew_high_reward_arm': rew_high_reward_arm,
            'rew_low_reward_arm': rew_low_reward_arm,
            'high_reward_arm': self.gt_arm,
            }
        self.trial.update(kwargs)
        self.obs = np.zeros((1, self.observation_space.shape[0]))
        self.gt = np.array([self.gt_arm])

    def _step(self, action):
        trial = self.trial
        info = {'new_trial': True, 'gt': self.gt}

        obs = self.obs[0]
        if action == trial['high_reward_arm']:
            reward = trial['rew_high_reward_arm']
            self.performance = 1
        else:
            reward = trial['rew_low_reward_arm']

        return obs, reward, False, info
