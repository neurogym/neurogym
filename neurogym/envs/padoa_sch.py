#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:10:35 2019

@author: molano

Economic choice task, based on

  Neurons in the orbitofrontal cortex encode economic value.
  C Padoa-Schioppa & JA Assad, Nature 2006.

  http://dx.doi.org/10.1038/nature04676

"""
from __future__ import division

import numpy as np
from neurogym.ops import tasktools
import neurogym as ngym
from gym import spaces


class PadoaSch(ngym.EpochEnv):
    metadata = {
        'description': 'Agents choose between two stimuli (A and B; where A is preferred) offered in different amounts.',
        'paper_link': 'https://www.nature.com/articles/nature04676',
        'paper_name': '''Neurons in the orbitofrontal cortex encode economic value''',
        'default_timing': {
            'fixation': ('constant', 1500),
            'offer_on': ('uniform', [1000, 2000]),
            'decision': ('constant', 750)},
    }

    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt, timing=timing)
        # Inputs
        self.inputs = tasktools.to_map('FIXATION', 'L-A', 'L-B', 'R-A',
                                       'R-B', 'N-L', 'N-R')

        # Actions
        self.actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT',
                                        'CHOOSE-RIGHT')

        # trial conditions
        self.A_to_B = 2.2
        self.juices = [('A', 'B'), ('B', 'A')]
        self.offers = [(0, 1), (1, 3), (1, 2), (1, 1), (2, 1),
                       (3, 1), (4, 1), (6, 1), (2, 0)]

        # Input noise
        self.sigma = np.sqrt(2*100*0.001)
        self.sigma_dt = self.sigma/np.sqrt(self.dt)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_MISS = 0.
        self.abort = False
        self.R_B = 0.1
        self.R_A = self.A_to_B * self.R_B

        # Increase initial policy -> baseline weights
        self.baseline_Win = 10

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7, ),
                                            dtype=np.float32)

    # Input scaling
    def scale(self, x):
        return x/5

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.trial = {
            'juice': self.rng.choice(self.juices),
            'offer': self.rng.choice(self.offers),
        }
        self.trial.update(kwargs)

        juiceL, juiceR = self.trial['juice']
        nB, nA = self.trial['offer']

        if juiceL == 'A':
            nL, nR = nA, nB
        else:
            nL, nR = nB, nA

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('offer_on', after='fixation')
        self.add_epoch('decision', after='offer_on', last_epoch=True)

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------
        self.set_ob('fixation', [1]+[0]*6)
        ob = self.view_ob('offer_on')
        ob[:, 0] = 1
        ob[:, self.inputs['L-'+juiceL]] = 1
        ob[:, self.inputs['R-'+juiceR]] = 1
        ob[:, self.inputs['N-L']] = self.scale(nL)
        ob[:, self.inputs['N-R']] = self.scale(nR)
        ob[:, [self.inputs['N-L'], self.inputs['N-R']]] += \
            np.random.randn(ob.shape[0], 2) * self.sigma_dt

    def _step(self, action):
        trial = self.trial

        new_trial = False

        obs = self.obs_now

        reward = 0
        if self.in_epoch('fixation') or self.in_epoch('offer_on'):
            if action != self.actions['FIXATE']:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action in [self.actions['CHOOSE-LEFT'],
                          self.actions['CHOOSE-RIGHT']]:
                new_trial = True

                juiceL, juiceR = trial['juice']

                nB, nA = trial['offer']
                rA = nA * self.R_A
                rB = nB * self.R_B

                if juiceL == 'A':
                    rL, rR = rA, rB
                else:
                    rL, rR = rB, rA

                if action == self.actions['CHOOSE-LEFT']:
                    reward = rL
                elif action == self.actions['CHOOSE-RIGHT']:
                    reward = rR

        return obs, reward, False, {'new_trial': new_trial, 'gt': 0}
