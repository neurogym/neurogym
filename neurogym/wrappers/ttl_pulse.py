#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:10:01 2020

@author: martafradera
"""

from gym.core import Wrapper


class TTLPulse(Wrapper):
    metadata = {
        'description': 'Outputs extra pulses that will be non-zero during ' +
        'specified periods.',
        'paper_link': None,
        'paper_name': None,
        'periods': 'List of list specifying the on periods for each pulse. ' +
        '(def: [])'
    }

    def __init__(self, env, periods=[]):
        super().__init__(env)

        self.periods = periods

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for ind_p, periods in enumerate(self.periods):
            info['signal_' + str(ind_p)] = 0
            for per in periods:
                if self.env.in_period(per):
                    info['signal_' + str(ind_p)] = 1

        return obs, reward, done, info
