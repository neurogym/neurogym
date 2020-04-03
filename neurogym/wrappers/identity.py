#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 07:11:15 2020

@author: manuel
"""

import neurogym as ngym


class Identity(ngym.TrialWrapper):
    """It does nothing"""
    metadata = {
    }

    def __init__(self, env, id_='0'):
        super().__init__(env)
        self.id = id_

    def new_trial(self, **kwargs):
        print('----------------')
        print('wrapper new_trial ', self.id)
        # ntr_fn = new_tr_fn or self.new_trial
        self.env.new_trial(**kwargs)

    def step(self, action, new_tr_fn=None):
        print('wrapper step', self.id)
        ntr_fn = new_tr_fn or self.new_trial
        obs, reward, done, info = self.env.step(action, new_tr_fn=ntr_fn)
        return obs, reward, done, info
