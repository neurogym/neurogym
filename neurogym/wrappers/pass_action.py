#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gym import Wrapper
from gym import spaces


class PassAction(Wrapper):
    """Modifies observation by adding the previous action."""
    metadata = {
        'description': 'Modifies observation by adding the previous action.',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # TODO: This is not adding one-hot
        env_oss = env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(env_oss+1,),
                                            dtype=np.float32)
    def reset(self, step_fn=None):
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(step_fn=step_fn)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs, np.array([action])))
        return obs, reward, done, info
