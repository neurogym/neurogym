#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-arm Bandit task
TODO: add the actual papers
"""
import numpy as np

import neurogym as ngym
from neurogym import spaces


class Bandit(ngym.TrialEnv):
    """Multi-arm bandit task.

    On each trial, the agent is presented with multiple choices. Each
    option produces a reward of a certain magnitude given a certain probability.

    Args:
        n: int, the number of choices (arms)
        p: tuple of length n, describes the probability of each arm
            leading to reward
        rewards: tuple of length n, describe the reward magnitude of each option when rewarded
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41593-018-0147-8',
        'paper_name': 'Prefrontal cortex as a meta-reinforcement learning' +
        ' system',
        'tags': ['n-alternative']
    }

    def __init__(self, dt=100, n=2, p=(.5, .5), rewards=None, timing=None):
        super().__init__(dt=dt)
        if timing is not None:
            print('Warning: Bandit task does not require timing variable.')

        if rewards:
            self.rewards = rewards
        else:
            self.rewards = np.ones(n)  # 1 for every arm

        self.n = n
        self.p = np.array(p)  # Reward probabilities

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(n)

    def _new_trial(self, **kwargs):
        trial = {'p': self.p, 'rewards': self.rewards}
        trial.update(kwargs)

        self.ob = np.zeros((1, self.observation_space.shape[0]))

        return trial

    def _step(self, action):
        trial = self.trial

        ob = self.ob[0]
        reward = (self.rng.random() < trial['p'][action]) * trial['rewards'][action]
        info = {'new_trial': True}

        return ob, reward, False, info
