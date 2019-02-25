#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 08:52:21 2019

@author: molano
"""

import gym
import numpy as np
from gym.utils import seeding


class ngym(gym.Env):
    """
    two-alternative forced choice task where the probability of repeating the
    previous choice is parametrized
    """
    def __init__(self, dt=0.1):
        super().__init__()
        self.dt = dt  # TODO: revisit, ms or s?
        self.t = 0
        self.num_tr = 0
        self.rng = np.random.RandomState(seed=0)
        self.perf = 0

    def step(self, action):
        """
        receives an action and returns a new state, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information
        """
        pass

    def reset(self):
        """
        restarts the experiment with the same parameters
        """
        print('reset --------------------------')
        print('mean performnace: ' + str(self.perf))
        self.num_tr += 1
        self.trial = self._new_trial(self.rng, self.dt)
        self.t = 0
        obs, _, _, _ = self.step(self.action_space.sample())
        return obs

    def render(self, mode='human'):
        """
        plots relevant variables/parameters
        """
        pass

    # Auxiliary functions
    def seed(self, seed=None):  # TODO: what is this function for?
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _new_trial(self):
        """
        starts a new trials within the current experiment
        """
        pass

    def analysis():
        """
        performs behavioral analysis relevant for the task
        (i.e. psychometric cuves)
        """
        pass
