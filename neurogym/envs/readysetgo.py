#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:48:19 2019

@author: molano


Ready-Set-Go task and Contextual Ready-Set-Go task, based on

  Flexible Sensorimotor Computations through Rapid
  Reconfiguration of Cortical Dynamics
  Evan D. Remington, Devika Narain,
  Eghbal A. Hosseini, Mehrdad Jazayeri, Neuron 2018.

  https://www.cell.com/neuron/pdf/S0896-6273(18)30418-5.pdf

"""
from __future__ import division

import numpy as np
from gym import spaces
from neurogym.ops import tasktools
import neurogym as ngym


def get_default_timing():
    return {'fixation': ('constant', 500),
            'ready': ('constant', 83),
            'measure': ('choice', [500, 580, 660, 760, 840, 920, 1000]),
            'set': ('constant', 83)}


class ReadySetGo(ngym.EpochEnv):
    def __init__(self, dt=80, timing=None, gain=1):
        super().__init__(dt=dt)
        # if dt > 80:
        # raise ValueError('dt {:0.2f} too large for this task.'.format(dt))
        # Actions (fixate, go)
        self.actions = [-1, 1]
        # Input noise
        self.sigma = np.sqrt(2*100*0.01)
        # gain
        self.gain = gain

        default_timing = get_default_timing()
        if timing is not None:
            default_timing.update(timing)
        self.set_epochtiming(default_timing)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False
        # set action and observation space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        self.add_epoch('fixation', after=0)
        self.add_epoch('ready', after='fixation')
        self.add_epoch('measure', after='fixation')
        self.add_epoch('set', after='measure')
        measure = self.measure_1 - self.measure_0
        production = measure * self.gain
        self.add_epoch('production', duration=2*production, after='set', last_epoch=True)

        self.set_ob('fixation', [1, 0, 0])
        self.set_ob('ready', [0, 1, 0])
        self.set_ob('set', [0, 0, 1])

        return {
            'measure': measure,
            'production': production,
            }

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'new_trial': False, 'gt': np.zeros((2,))}
        reward = 0
        obs = self.obs[self.t_ind]
        if self.in_epoch('fixation'):
            info['gt'][0] = 1
            if self.actions[action] != -1:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        if self.in_epoch('production'):
            t_prod = self.t - self.measure_1  # time from end of measure
            eps = abs(t_prod - trial['production'])
            if eps < self.dt/2 + 1:
                info['gt'][1] = 1
            else:
                info['gt'][0] = 1
            if action == 1:
                info['new_trial'] = True  # terminate
                # actual production time
                eps_threshold = 0.2*trial['production']+25
                if eps > eps_threshold:
                    reward = self.R_FAIL
                else:
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = min(reward, 0.1)
                    reward *= self.R_CORRECT
        else:
            info['gt'][0] = 1

        return obs, reward, False, info
