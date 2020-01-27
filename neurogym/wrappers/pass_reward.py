#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:25:12 2019

@author: linux

@author: molano
"""

import numpy as np
from gym.core import Wrapper
from gym import spaces


class PassReward(Wrapper):
    metadata = {
        'description': """Modifies observation by adding the previous
        reward.""",
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env):
        super().__init__(env)
        env_oss = env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(env_oss+1,),
                                            dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        return np.concatenate((obs, np.array([0])))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs, np.array([reward])))
        return obs, reward, done, info
