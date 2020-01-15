#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ready-set-go task."""

from __future__ import division

import numpy as np
from gym import spaces
from neurogym.ops import tasktools
import neurogym as ngym


class ReadySetGo(ngym.EpochEnv):
    metadata = {
        'paper_link': 'https://www.sciencedirect.com/science/article/pii/S0896627318304185',
        'paper_name': '''Flexible Sensorimotor Computations through Rapid
        Reconfiguration of Cortical Dynamics''',
        'default_timing': {
            'fixation': ('constant', 500),
            'ready': ('constant', 83),
            'measure': ('choice', [500, 580, 660, 760, 840, 920, 1000]),
            'set': ('constant', 83)},
    }

    def __init__(self, dt=80, timing=None, gain=1):
        super().__init__(dt=dt, timing=timing)

        self.sigma = np.sqrt(2*100*0.01)
        self.gain = gain

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False
        # set action and observation space
        self.action_space = spaces.Discrete(2)  # (fixate, go)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        measure = (self.timing_fn['measure']() // self.dt) * self.dt
        self.trial = {
            'measure': measure,
            'gain': self.gain
        }
        self.trial.update(kwargs)

        self.trial['production'] = measure * self.trial['gain']

        self.add_epoch('fixation', after=0)
        self.add_epoch('ready', after='fixation')
        self.add_epoch('measure', duration=measure, after='fixation')
        self.add_epoch('set', after='measure')
        self.add_epoch('production', duration=2*self.trial['production'],
                       after='set', last_epoch=True)

        self.set_ob('fixation', [1, 0, 0])
        self.set_ob('ready', [0, 1, 0])
        self.set_ob('set', [0, 0, 1])

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        reward = 0
        obs = self.obs_now
        gt = np.zeros((2,))
        gt[0] = 1
        new_trial = False
        if self.in_epoch('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        if self.in_epoch('production'):
            t_prod = self.t - self._end_t['measure']  # time from end of measure
            eps = abs(t_prod - trial['production'])
            if eps < self.dt/2 + 1:
                gt[1] = 1

            if action == 1:
                new_trial = True  # terminate
                # actual production time
                eps_threshold = 0.2*trial['production']+25
                if eps > eps_threshold:
                    reward = self.R_FAIL
                else:
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = min(reward, 0.1)
                    reward *= self.R_CORRECT

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


class MotorTiming(ngym.EpochEnv):
    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41593-017-0028-6',
        'paper_name': '''Flexible timing by temporal scaling of cortical responses''',
        'default_timing': {
            'fixation': ('constant', 500),
            'cue': ('truncated_exponential', [1000, 500, 1500]),
            'set': ('constant', 50)}
    }

    def __init__(self, dt=80, timing=None, gain=1):
        super().__init__(dt=dt, timing=timing)

        self.sigma = np.sqrt(2*100*0.01)
        self.gain = gain
        self.production_ind = [0, 1]
        self.intervals = [800, 1500]

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False
        # set action and observation space
        self.action_space = spaces.Discrete(2)  # (fixate, go)
        # Fixation, Interval indicator x2, Set
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        self.trial = {
            'production_ind': self.rng.choice(self.production_ind)
        }
        self.trial.update(kwargs)

        self.trial['production'] = self.intervals[self.trial['production_ind']]

        self.add_epoch('fixation', after=0)
        self.add_epoch('cue', after='fixation')
        self.add_epoch('set', after='cue')
        self.add_epoch('production', duration=2*self.trial['production'],
                       after='set', last_epoch=True)

        self.set_ob('fixation', [1, 0, 0, 0])
        ob = self.view_ob('cue')
        ob[:, 0] = 1
        ob[:, self.trial['production_ind']+1] = 1
        ob = self.view_ob('set')
        ob[:, 0] = 1
        ob[:, self.trial['production_ind'] + 1] = 1
        ob[:, 3] = 1


    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        reward = 0
        obs = self.obs_now
        gt = np.zeros((2,))
        gt[0] = 1
        new_trial = False
        if self.in_epoch('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        if self.in_epoch('production'):
            t_prod = self.t - self._end_t['set']  # time from end of measure
            eps = abs(t_prod - trial['production'])
            if eps < self.dt/2 + 1:
                gt[1] = 1

            if action == 1:
                new_trial = True  # terminate
                # actual production time
                eps_threshold = 0.2*trial['production']+25
                if eps > eps_threshold:
                    reward = self.R_FAIL
                else:
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = min(reward, 0.1)
                    reward *= self.R_CORRECT

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}