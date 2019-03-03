#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:48:19 2019

@author: molano


Perceptual decision-making task, based on

  Bounded integration in parietal cortex underlies decisions even when viewing
  duration is dictated by the environment.
  R Kiani, TD Hanks, & MN Shadlen, JNS 2008.

  http://dx.doi.org/10.1523/JNEUROSCI.4761-07.2008

"""
from __future__ import division

import numpy as np
from gym import spaces
import tasktools
import ngym


class RDM(ngym.ngym):
    # Inputs
    inputs = tasktools.to_map('FIXATION', 'LEFT', 'RIGHT')

    # Actions
    actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT')

    # Trial conditions
    left_rights = [-1, 1]
    cohs = [0, 6.4, 12.8, 25.6, 51.2]  # Easier: [25.6, 51.2, 102.4, 204.8]

    # Input noise
    sigma = np.sqrt(2*100*0.01)

    # Durations
    fixation = 750
    stimulus_min = 80
    stimulus_mean = 330
    stimulus_max = 1500
    decision = 500
    tmax = fixation + stimulus_max + decision

    # Rewards
    R_ABORTED = -1.
    R_CORRECT = +1.
    R_FAIL = 0.
    R_MISS = 0.
    abort = False

    def __init__(self, dt=100):
        super().__init__(dt=dt)
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, ),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.trial = self._new_trial()
        print('------------------------')
        print('RDM task')
        print('time step: ' + str(self.dt))
        print('------------------------')

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        stimulus = tasktools.truncated_exponential(self.rng, self.dt,
                                                   self.stimulus_mean,
                                                   xmin=self.stimulus_min,
                                                   xmax=self.stimulus_max)

        # TODO: don't align, not PEP8
        durations = {
            'fix_grace': (0, 100),
            'fixation':  (0, self.fixation),
            'stimulus':  (self.fixation, self.fixation + stimulus),
            'decision':  (self.fixation + stimulus,
                          self.fixation + stimulus + self.decision),
            'tmax':      self.tmax
            }

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        ground_truth = tasktools.choice(self.rng, self.left_rights)

        coh = tasktools.choice(self.rng, self.cohs)

        return {
            'durations':   durations,
            'ground_truth':  ground_truth,
            'coh':         coh
            }

    # Input scaling
    def scale(self, coh):
        return (1 + coh/100)/2

    def step(self, action):
        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'continue': True}
        reward = 0
        tr_perf = False
        # if self.t not in epochs['decision']:
        if not self.in_epoch(self.t, 'decision'):
            if (action != self.actions['FIXATE'] and
                    not self.in_epoch(self.t, 'fix_grace')):
                info['continue'] = not self.abort
                reward = self.R_ABORTED
        else:  # elif self.t in epochs['decision']:
            # TODO: this part can be simplified
            if action == self.actions['CHOOSE-LEFT']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'L'
                info['t_choice'] = self.t
                info['correct'] = (trial['ground_truth'] < 0)
                if info['correct']:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL
            elif action == self.actions['CHOOSE-RIGHT']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'R'
                info['t_choice'] = self.t
                info['correct'] = (trial['ground_truth'] > 0)
                if info['correct']:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL
        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        if trial['ground_truth'] < 0:
            high = self.inputs['LEFT']
            low = self.inputs['RIGHT']
        else:
            high = self.inputs['RIGHT']
            low = self.inputs['LEFT']

        obs = np.zeros(len(self.inputs))
        # if self.t in epochs['fixation'] or self.t in epochs['stimulus']:
        if self.in_epoch(self.t, 'fixation') or\
           self.in_epoch(self.t, 'stimulus'):
            obs[self.inputs['FIXATION']] = 1
        # if self.t in epochs['stimulus']:
        if self.in_epoch(self.t, 'stimulus'):
            obs[high] = self.scale(+trial['coh']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
            obs[low] = self.scale(-trial['coh']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['continue'],
                                                self.R_MISS, reward)

        if new_trial:
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward, self.num_tr,
                                       self.num_tr_exp, self.num_tr_perf, tr_perf)
            self.trial = self._new_trial()
        else:
            self.t += self.dt

        self.store_data(obs, action, reward)
        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8
