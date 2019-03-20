#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:25:12 2019

@author: linux

@author: molano
"""

from gym.core import Wrapper
import numpy as np


class PassReward(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs, np.array([reward])))
        return obs, reward, done, info
