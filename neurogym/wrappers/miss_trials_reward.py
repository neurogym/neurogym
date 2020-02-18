#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:58:46 2020

@author: molano
"""
import neurogym as ngym


class MissTrialReward(ngym.TrialWrapper):
    metadata = {
        'description': 'Add a negative reward if a trial ends with no action.',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env, r_miss=0.):
        """
        r_miss: Reward given when a miss trial occurs.(def: 0, int)
        """
        super().__init__(env)
        self.r_miss = r_miss

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['trial_endwith_tmax']:
            reward += self.r_miss
        return obs, reward, done, info
