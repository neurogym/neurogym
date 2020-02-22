#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:53:05 2019

@author: linux

@author: molano
"""

import numpy as np
import neurogym as ngym
from gym import spaces


class PassAction(ngym.TrialWrapper):
    metadata = {
        'description': 'Modifies observation by adding the previous action.',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env):
        """
        Modifies observation by adding the previous action.
        """
        super().__init__(env)
        self.env = env
        # TODO: This is not adding one-hot
        env_oss = env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(env_oss+1,),
                                            dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        return np.concatenate((obs, np.array([0])))

    def step(self, action):
        # TODO: Need to turn action into one-hot
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs, np.array([action])))
        return obs, reward, done, info
