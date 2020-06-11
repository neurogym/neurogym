#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from neurogym.core import TrialWrapperV2
from gym import spaces


class PassAction(TrialWrapperV2):
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

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs, np.array([action])))
        return obs, reward, done, info
