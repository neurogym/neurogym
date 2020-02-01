#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:45:32 2019

@author: molano
"""
from gym.core import Wrapper
import numpy as np


class Noise(Wrapper):
    metadata = {
        'description': '''Add Gaussian noise to the observations.''',
        'paper_link': None,
        'paper_name': None,
        'std_noise': 'Standard deviation of noise. (def: 0.1)',
    }

    def __init__(self, env, std_noise=.1):
        super().__init__(env)
        self.env = env
        self.std_noise = std_noise
        self.std_noise = self.std_noise / self.env.dt

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs += np.random.normal(loc=0, scale=self.std_noise, size=obs.shape)
        return obs, reward, done, info
