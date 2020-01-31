#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:10:01 2020

@author: martafradera
"""

import numpy as np
from gym.core import Wrapper
from gym import spaces


class TTLPulse(Wrapper):
    metadata = {
        'description': '',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env, epochs=[['fixation', 'stimulus', 'delay',
                                     'decision']]):
        super().__init__(env)

        self.epochs = epochs

        self.env_oss = env.observation_space.shape[0]
        num_types = self.epochs.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.env_oss+num_types,),
                                            dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        return np.concatenate((obs, np.array([0])))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        for case in self.epochs:
            if self.in_epoch('fixation'):
                if 'fixation' in case:
                    obs = np.concatenate((obs, np.array([1])))
                else:
                    obs = np.concatenate((obs, np.array([0])))
            if self.in_epoch('stimulus'):
                if 'stimulus' in case:
                    obs = np.concatenate((obs, np.array([1])))
                else:
                    obs = np.concatenate((obs, np.array([0])))
            if self.in_epoch('delay'):
                if 'delay' in case:
                    obs = np.concatenate((obs, np.array([1])))
                else:
                    obs = np.concatenate((obs, np.array([0])))
            if self.in_epoch('decision'):
                if 'decision' in case:
                    obs = np.concatenate((obs, np.array([1])))
                else:
                    obs = np.concatenate((obs, np.array([0])))

        return obs, reward, done, info
    
