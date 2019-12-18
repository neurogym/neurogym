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
        trial = self.env._new_trial()
        trial['durations']['decision'] = (trial['durations']['fixation'][1],
                                          trial['durations']['decision'][1])
        return trial

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env._step(action)

        if info['new_trial']:
            self.env.trial = self._new_trial()

        return obs, reward, done, info
