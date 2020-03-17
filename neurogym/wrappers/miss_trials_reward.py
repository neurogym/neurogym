#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import neurogym as ngym


class MissTrialReward(ngym.TrialWrapper):
    """Provide reward if trial is miss.

    Args:
        r_miss: Reward given when a miss trial occurs.(def: 0, int)
    """
    metadata = {
        'description': 'Add a negative reward if a trial ends with no action.',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env, r_miss=0.):
        super().__init__(env)
        self.r_miss = r_miss

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['trial_endwith_tmax']:
            reward += self.r_miss
        return obs, reward, done, info
