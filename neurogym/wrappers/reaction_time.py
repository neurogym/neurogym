#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:07:21 2019

@author: molano
"""

import neurogym as ngym


class ReactionTime(ngym.TrialWrapper):
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

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action, new_tr_fn=None):
        assert 'stimulus' in self.env.start_t.keys(),\
            'Reaction time wrapper requires a stimulus period'
        if self.t_ind == 0:
            self.env.start_t['decision'] = self.env.start_t['stimulus']
        ntr_fn = new_tr_fn or self.new_trial
        obs, reward, done, info = self.env.step(action, new_tr_fn=ntr_fn)
        return obs, reward, done, info
