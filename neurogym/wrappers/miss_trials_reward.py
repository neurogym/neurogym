#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:58:46 2020

@author: molano
"""
from gym.core import Wrapper


class MissTrialReward(Wrapper):
    metadata = {
        'description': 'Add a negative reward if a trial ends with no action.',
        'paper_link': None,
        'paper_name': None,
        'r_miss': 'Reward given when a miss trial occurs.(def: 0)',
    }

    def __init__(self, env, r_miss=0.):
        super().__init__(env)
        self.r_miss = r_miss

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['trial_endwith_tmax']:
            reward += self.r_miss
        return obs, reward, done, info
