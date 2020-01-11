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
    def __init__(self, dt=100, timing=(750, 1000, 2000, 750)):
        # call ngm __init__ function
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

        # Durations
        self.fixation = timing[0]
        self.offer_on_min = timing[1]
        self.offer_on_max = timing[2]
        self.offer_on_mean = (self.offer_on_min + self.offer_on_max) / 2
        self.decision = timing[3]
        self.mean_trial_duration = self.fixation + self.offer_on_mean +\
            self.decision

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

        self.seed()
        self.viewer = None

    def __str__(self):
        string = 'mean trial duration: ' + str(self.mean_trial_duration) + '\n'
        string += 'max num. steps: ' + str(self.mean_trial_duration / self.dt)
        return string

    # Input scaling
    def scale(self, x):
        return x/5

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        offer_on = tasktools.uniform(self.rng, self.dt, self.offer_on_min,
                                     self.offer_on_max)

        self.add_epoch('fixation', self.fixation, start=0)
        self.add_epoch('offer-on', offer_on, after='fixation')
        self.add_epoch('decision', self.decision, after='offer-on', last_epoch=True)

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

        return {
            'juice':     juice,
            'offer':     offer,
            'nL':        nL,
            'nR':        nR
            }

    def _step(self, action):
        trial = self.trial

        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        info = {'new_trial': False}
        info['gt'] = np.zeros((3,))
        reward = 0
        if (self.in_epoch('fixation') or
                self.in_epoch('offer-on')):
            if (action != self.actions['FIXATE']):
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

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------
        obs = np.zeros(len(self.inputs))
        if not self.in_epoch('decision'):
            obs[self.inputs['FIXATION']] = 1
        if self.in_epoch('offer-on'):
            juiceL, juiceR = trial['juice']
            obs[self.inputs['L-'+juiceL]] = 1
            obs[self.inputs['R-'+juiceR]] = 1

            obs[self.inputs['N-L']] = self.scale(trial['nL']) +\
                self.rng.gauss(mu=0, sigma=self.sigma)/np.sqrt(self.dt)
            obs[self.inputs['N-R']] = self.scale(trial['nR']) +\
                self.rng.gauss(mu=0, sigma=self.sigma)/np.sqrt(self.dt)

        return obs, reward, False, info
