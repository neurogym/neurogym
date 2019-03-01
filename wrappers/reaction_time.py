#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:07:21 2019

@author: molano
"""

from gym.core import Wrapper


class ReactionTime(Wrapper):
    """
    modfies a given environment by allowing the network to act at any time
    after the fixation period
    """
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.env = env

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        trial = self.env._new_trial()  #TODO: error
        trial['durations']['decision'] = (trial['durations']['fixation'],
                                          trial['durations']['decision'][1])

        return trial
