#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import numpy as np


class Noise(gym.Wrapper):
    """Add Gaussian noise to the observations.

    Args:
        std_noise: Standard deviation of noise. (def: 0.1)
        perf_th: If != None, the wrapper will adjust the noise so the mean
            performance is not larger than perf_th. (def: None, float)
        w: Window used to compute the mean performance. (def: 100, int)
        step_noise: Step used to increment/decrease std. (def: 0.001, float)
    """
    metadata = {
        'description': 'Add Gaussian noise to the observations.',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env, std_noise=.1, perf_th=None, w=200,
                 step_noise=0.0001):
        super().__init__(env)
        self.env = env
        self.std_noise = std_noise
        self.std_noise = self.std_noise / self.env.dt
        self.init_noise = 0
        self.step_noise = step_noise
        self.w = w
        self.perf_th = perf_th
        if self.perf_th is not None:
            self.perf_th = perf_th
            self.perf = []
            self.std_noise = 0

    def reset(self, step_fn=None):
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(step_fn=step_fn)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # adjust noise
        if self.perf_th is not None and info['new_trial']:
            assert 'performance' in info, 'Adjusting noise is only possible' +\
                ' with task that output a performance value'
            self.perf.append(info['performance'])
            self.min_w = len(self.perf) > self.w
            if self.min_w:
                self.perf.pop(0)

            perf_mean = np.mean(self.perf)
            if perf_mean > self.perf_th and self.min_w:
                self.std_noise += self.step_noise
            elif perf_mean < self.perf_th and self.std_noise > 0:
                self.std_noise = max(0, self.std_noise-self.step_noise)
            info['perf_mean'] = perf_mean
            info['std_noise'] = self.std_noise

        # add noise
        obs += self.env.rng.normal(loc=0, scale=self.std_noise,
                                   size=obs.shape)
        return obs, reward, done, info
