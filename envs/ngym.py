#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 08:52:21 2019

@author: molano
"""

import gym
import random


class ngym(gym.Env):
    """
    neurogym superclass specifying the basic structure of all tasks
    """
    def __init__(self, dt=100):
        super().__init__()
        self.dt = dt
        self.t = 0
        self.num_tr = 0
        self.perf = 0
        self.num_tr_perf = 0
        self.num_tr_exp = 10000000  # num trials after which done = True
        self.seed()

    def step(self, action):
        """
        receives an action and returns a new state, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information (info). Aditionally, if the current trial is done
        (info['new_trial']==True) it calls the function _new_trial.
        """
        return None, None, None, None

    def reset(self):
        """
        restarts the experiment with the same parameters
        """
        if self.num_tr > 1:
            print(self.__class__.__name__)
            print('percentage of trials performed: ' +
                  str(100*self.num_tr_perf/self.num_tr_exp))
            print('mean performance: ' + str(self.perf))

        self.perf = 0
        self.num_tr_perf = 0
        self.num_tr = 1
        self.t = 0

        self.trial = self._new_trial()
        obs, _, _, _ = self.step(self.action_space.sample())
        return obs

    def render(self, mode='human'):
        """
        plots relevant variables/parameters
        """
        pass

    # Auxiliary functions
    def seed(self, seed=None):
        self.rng = random
        self.rng.seed(seed)
        return [seed]

    def _step(self, action):
        """
        receives an action and returns a new state, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information
        """
        return None, None, None, None

    def _new_trial(self):
        """Starts a new trial within the current experiment.

        Returns:
            trial_info: a dictionary of trial information
        """
        return {}

    def in_epoch(self, t, epoch):
        """Check if t is in epoch."""
        dur = self.trial['durations']
        return (dur[epoch][0] <= t < dur[epoch][1])
