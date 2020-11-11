#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gym import Wrapper
from neurogym import spaces

class PassAction(Wrapper):
    """Modifies observation by adding the previous action."""
    metadata = {
        'description': 'Modifies observation by adding the previous action as one-hot encoded vector',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        env_oss = env.observation_space.shape[0]
        self.num_act = env.action_space.n
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(env_oss+self.num_act,),
                                            dtype=np.float32,name = self.ob_dict)
        self.action_space = spaces.Discrete(self.num_act,name = self.act_dict) #without this line, env.action_space becomes None for some reason
    def reset(self, step_fn=None):
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(step_fn=step_fn)

    def step(self, action):
        actions = np.zeros(self.num_act)
        actions[action] = 1
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs, actions))
        return obs, reward, done, info

