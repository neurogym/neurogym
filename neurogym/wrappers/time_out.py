#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:49:36 2020

@author: molano
"""
from neurogym.core import TrialWrapper


class TimeOut(TrialWrapper):  # TODO: Make this a trial wrapper instead?
    """Allow reaction time response.

    Modifies a given environment by allowing the network to act at
    any time after the fixation period.
    """
    metadata = {
        'description': 'Modifies a given environment by allowing the network' +
        ' to act at any time after the fixation period.',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env, time_out=0):
        super().__init__(env)
        self.env = env
        self.time_out = time_out

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        next_fix = None if self.performance == 1 else self.time_out
        kwargs.update({'fixation': next_fix})
        self.env.new_trial(**kwargs)
