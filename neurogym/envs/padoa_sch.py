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


def get_default_timing():
    return {'fixation': ('constant', 750),
            'offer_on': ('truncated_exponential', [1500, 1000, 2000]),
            'decision': ('constant', 750)}


class PadoaSch(ngym.EpochEnv):
    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt)
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

        default_timing = get_default_timing()
        if timing is not None:
            default_timing.update(timing)
        self.set_epochtiming(default_timing)

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
        juice = self.rng.choice(self.juices)

        offer = self.rng.choice(self.offers)

        juiceL, juiceR = juice
        nB, nA = offer

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
        tmp = [1]+[0]*6
        tmp[self.inputs['L-'+juiceL]] = 1
        tmp[self.inputs['R-'+juiceR]] = 1
        tmp[self.inputs['N-L']] = self.scale(nL)
        tmp[self.inputs['N-R']] = self.scale(nR)
        self.set_ob('offer_on', tmp)
        self.obs[self.offer_on_ind0:self.offer_on_ind1, [self.inputs['N-L'], self.inputs['N-R']]] += \
        np.random.randn(self.offer_on_ind1-self.offer_on_ind0, 2) * (self.sigma/np.sqrt(self.dt))

        return {
            'juice': juice,
            'offer': offer,
            'nL': nL,
            'nR': nR
            }

    def _step(self, action):
        trial = self.trial

        info = {'new_trial': False}

        obs = self.obs_now

        reward = 0
        if self.in_epoch('fixation') or self.in_epoch('offer_on'):
            if action != self.actions['FIXATE']:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action in [self.actions['CHOOSE-LEFT'],
                          self.actions['CHOOSE-RIGHT']]:
                info['new_trial'] = True

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

        return obs, reward, False, info
