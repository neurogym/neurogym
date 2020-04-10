#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:10:01 2020

@author: martafradera
"""

import neurogym as ngym


class TTLPulse(ngym.TrialWrapper):
    """Outputs extra pulses that will be non-zero during specified periods.

    Args:
        periods: List of list specifying the on periods for each pulse.
            (def: [], list)
    """
    metadata = {
        'description': 'Outputs extra pulses that will be non-zero during ' +
        'specified periods.',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env, periods=[]):
        super().__init__(env)

        self.periods = periods

    def step(self, action, new_tr_fn=None):
        info_tmp = {}
        for ind_p, periods in enumerate(self.periods):
            info_tmp['signal_' + str(ind_p)] = 0
            for per in periods:
                if self.env.in_period(per):
                    info_tmp['signal_' + str(ind_p)] = 1

        ntr_fn = new_tr_fn or self.new_trial
        obs, reward, done, info = self.env.step(action, new_tr_fn=ntr_fn)
        info.update(info_tmp)
        return obs, reward, done, info
