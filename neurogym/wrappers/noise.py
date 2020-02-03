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
        'description': 'Add Gaussian noise to the observations.',
        'paper_link': None,
        'paper_name': None,
        'std_noise': 'Standard deviation of noise. (def: 0.1)',
        'w': 'Window length. (def: 100)'
    }

    def __init__(self, env, std_noise=.1, rew_th=None, w=10, step_noise=0.01):
        super().__init__(env)
        self.env = env
        self.std_noise = std_noise
        self.std_noise = self.std_noise / self.env.dt
        self.init_noise = 0
        self.step_noise = step_noise
        self.w = w
        self.min_w = False
        self.rew_th = rew_th
        if self.rew_th is not None:
            self.rew_th = rew_th
            self.rewards = []
            self.std_noise = 0
        else:
            self.min_w = True
            self.init_noise = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['std_noise'] = self.std_noise
        if info['new_trial']:
            if self.rew_th is not None:
                self.rewards.append(reward)
                if len(self.rewards) > self.w:
                    self.rewards.pop(0)
                    self.min_w = True

                rew_mean = np.mean(self.rewards)
                info['rew_mean'] = rew_mean
                if rew_mean > self.rew_th and self.min_w is True:
                    self.std_noise += self.step_noise

        # add noise
        obs += np.random.normal(loc=0, scale=self.std_noise,
                                size=obs.shape)

        return obs, reward, done, info
