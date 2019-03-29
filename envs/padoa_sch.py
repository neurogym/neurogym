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
import ngym
from gym import spaces
import tasktools


class PadoaSch(ngym.ngym):
    def __init__(self, dt=100):
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
        self.fixation = 750
        self.offer_on_min = 1000
        self.offer_on_max = 2000
        self.offer_on_mean = (self.offer_on_min + self.offer_on_max) / 2
        self.decision = 750
        self.mean_trial_duration = self.fixation + self.offer_on_mean +\
            self.decision
        print('mean trial duration: ' + str(self.mean_trial_duration) +
              ' (max num. steps: ' + str(self.mean_trial_duration/self.dt) +
              ')')
        # Rewards
        self.R_ABORTED = -1.
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

        self.trial = self._new_trial()

    # Input scaling
    def scale(self, x):
        return x/5

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        offer_on = tasktools.uniform(self.rng, self.dt, self.offer_on_min,
                                     self.offer_on_max)
        # maximum duration of current trial
        self.tmax = self.fixation + offer_on + self.decision
        durations = {
            'fixation':    (0, self.fixation),
            'offer-on':    (self.fixation, self.fixation + offer_on),
            'decision':    (self.fixation + offer_on, self.tmax),
            }

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        juice = tasktools.choice(self.rng, self.juices)

        offer = tasktools.choice(self.rng, self.offers)

        juiceL, juiceR = juice
        nB, nA = offer

        if juiceL == 'A':
            nL, nR = nA, nB
        else:
            nL, nR = nB, nA

        return {
            'durations': durations,
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

        # epochs = trial['epochs']
        info = {'continue': True}
        reward = 0
        tr_perf = False
        if (self.in_epoch(self.t, 'fixation') or
                self.in_epoch(self.t, 'offer-on')):
            if (action != self.actions['FIXATE']):
                info['continue'] = not self.abort
                reward = self.R_ABORTED
        if self.in_epoch(self.t, 'decision'):
            if action in [self.actions['CHOOSE-LEFT'],
                          self.actions['CHOOSE-RIGHT']]:
                tr_perf = True
                info['continue'] = False
                info['t_choice'] = self.t

                juiceL, juiceR = trial['juice']

                nB, nA = trial['offer']
                rA = nA * self.R_A
                rB = nB * self.R_B

                if juiceL == 'A':
                    rL, rR = rA, rB
                else:
                    rL, rR = rB, rA

                if action == self.actions['CHOOSE-LEFT']:
                    if juiceL == 'A':
                        info['choice'] = 'A'
                    else:
                        info['choice'] = 'B'
                    info['correct'] = (rL >= rR)
                    reward = rL
                elif action == self.actions['CHOOSE-RIGHT']:
                    if juiceR == 'A':
                        info['choice'] = 'A'
                    else:
                        info['choice'] = 'B'
                    info['correct'] = (rR >= rL)
                    reward = rR

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------
        obs = np.zeros(len(self.inputs))
        if not self.in_epoch(self.t, 'decision'):
            obs[self.inputs['FIXATION']] = 1
        if self.in_epoch(self.t, 'offer-on'):
            juiceL, juiceR = trial['juice']
            obs[self.inputs['L-'+juiceL]] = 1
            obs[self.inputs['R-'+juiceR]] = 1

            obs[self.inputs['N-L']] = self.scale(trial['nL']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
            obs[self.inputs['N-R']] = self.scale(trial['nR']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['continue'],
                                                self.R_MISS, reward)

        if new_trial:
            info['new_trial'] = True
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward,
                                       self.num_tr_perf, tr_perf)
        else:
            self.t += self.dt

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info, new_trial

    def step(self, action):
        obs, reward, done, info, new_trial = self._step(action)
        if new_trial:
            self.trial = self._new_trial()
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.95
