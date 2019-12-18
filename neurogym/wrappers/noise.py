#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:45:32 2019

@author: molano
"""
from gym.core import Wrapper
import numpy as np


class Noise(Wrapper):
    """
    modfies a given environment by adding gaussian noise to the observations
    """
    def __init__(self, env, std_noise=.1):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.std_noise = std_noise

    def step(self, action):
        obs, reward, done, info = self.env._step(action)
        obs += np.random.normal(loc=0, scale=self.std_noise, size=obs.shape)
        if info['new_trial']:
            self.env.trial = self._new_trial()

        return obs, reward, done, info
