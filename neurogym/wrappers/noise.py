"""Noise wrapper.

Created on Thu Feb 28 15:07:21 2019

@author: molano
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym


class Noise(gym.Wrapper):
    """Add Gaussian noise to the observations.

    Args:
        std_noise: Standard deviation of noise. (def: 0.1)
        perf_th: If != None, the wrapper will adjust the noise so the mean
            performance is not larger than perf_th. (def: None, float)
        w: Window used to compute the mean performance. (def: 100, int)
        step_noise: Step used to increment/decrease std. (def: 0.001, float)

    """

    metadata: dict[str, str | None] = {  # noqa: RUF012
        "description": "Add Gaussian noise to the observations.",
        "paper_link": None,
        "paper_name": None,
    }

    def __init__(self, env, std_noise=0.1) -> None:
        super().__init__(env)
        self.env = env
        self.std_noise = std_noise

    def reset(self, options=None):
        step_fn = options.get("step_fn") if options else None
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(options={"step_fn": step_fn})

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # add noise
        obs += self.env.rng.normal(loc=0, scale=self.std_noise, size=obs.shape)
        return obs, reward, terminated, truncated, info
