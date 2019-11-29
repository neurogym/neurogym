#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:53:05 2019

@author: linux

@author: molano
"""

from gym.core import Wrapper
import numpy as np
from gym import spaces


class PassAction(Wrapper):
    """
    modfies a given observation by adding the previous action
    """
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.env = env
        env_oss = env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(env_oss+1,),
                                            dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        return np.concatenate((obs, np.array([0])))

    def _step(self, action):
        return self.env._step(action)

    def step(self, action):
        obs, reward, done, info = self._step(action)
        obs = np.concatenate((obs, np.array([action])))
        return obs, reward, done, info
