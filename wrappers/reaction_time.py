#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:07:21 2019

@author: molano
"""

from gym.core import Wrapper


class ReactionTime(Wrapper):
    """
    modfies a given environment by changing the starting point and duration
    of the decision period
    """
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.env = env

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        trial = self.env._new_trial(self.rng, self.dt)
        trial['duratpns']['decision'] = (trial['durations']['fixation'],
                                         trial['durations']['decision'][1])

        return trial
