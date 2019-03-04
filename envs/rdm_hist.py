#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:03:35 2019

@author: molano

two-alternative forced choice task with trial-to-trial correlations, based on

Response outcomes gate the impact of expectations on perceptual decisions
Ainhoa Hermoso-Mendizabal, Alexandre Hyafil, Pavel Ernesto Rueda-Orozco,
Santiago Jaramillo, David Robbe, Jaime de la Rocha
doi: https://doi.org/10.1101/433409

"""
from __future__ import division

import numpy as np
from gym import spaces
import tasktools
import ngym


class RDM_hist(ngym.ngym):
    # Inputs
    inputs = tasktools.to_map('FIXATION', 'LEFT', 'RIGHT')

    # Actions
    actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT')

    # Trial conditions
    left_rights = [-1, 1]
    cohs = [0, 6.4, 12.8, 25.6, 51.2]

    # Input noise
    sigma = np.sqrt(2*100*0.01)

    # Durations
    fixation = 750
    stimulus_min = 1000
    stimulus_mean = 1001
    stimulus_max = 1002
    decision = 500
    tmax = fixation + stimulus_max + decision

    # Rewards
    R_ABORTED = -1.
    R_CORRECT = +1.
    R_FAIL = 0.
    R_MISS = 0.
    abort = False

    def __init__(self, dt=100, rep_prob=(.2, .8), block_dur=200):
        super().__init__(dt=dt)
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, ),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        # repeating prob variables
        # prob. of repeating the stimuli in the positions of previous trial
        self.rep_prob = rep_prob
        # position of the first stimulus
        self.left_right_prev_trial = tasktools.choice(self.rng, [0, 1])
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.rng, [0, 1])
        # duration of block (in number oif trials)
        self.block_dur = block_dur

        self.trial = self._new_trial()

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        stimulus = tasktools.truncated_exponential(self.rng, self.dt,
                                                   self.stimulus_mean,
                                                   xmin=self.stimulus_min,
                                                   xmax=self.stimulus_max)

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
        if self.num_tr % self.block_dur == 0:
            self.curr_block = int(not self.curr_block)
        if self.left_right_prev_trial == -1:
            probs = (self.rep_prob[self.curr_block],
                     1-self.rep_prob[self.curr_block])
        else:
            probs = (1-self.rep_prob[self.curr_block],
                     self.rep_prob[self.curr_block])

        left_right = self.rng.choice(self.left_rights,
                                     p=probs)

        self.left_right_prev_trial = left_right

        coh = tasktools.choice(self.rng, self.cohs)

        return {
            'fix_grace': (0, 100),
            'durations':   durations,
            'left_right':  left_right,
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
        # epochs = trial['epochs']
        info = {'continue': True}
        info['gt'] = trial['left_right']
        reward = 0
        tr_perf = False
        if not self.in_epoch(self.t, 'decision'):
            if (action != self.actions['FIXATE'] and
                    not self.in_epoch(self.t, 'fix_grace')):
                info['continue'] = not self.abort
                reward = self.R_ABORTED
        else:
            if action == self.actions['CHOOSE-LEFT']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'L'
                info['t_choice'] = self.t
                info['correct'] = (trial['left_right'] < 0)
                if info['correct']:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL
            elif action == self.actions['CHOOSE-RIGHT']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'R'
                info['t_choice'] = self.t
                info['correct'] = (trial['left_right'] > 0)
                if info['correct']:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL
        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        if trial['left_right'] < 0:
            high = self.inputs['LEFT']
            low = self.inputs['RIGHT']
        else:
            high = self.inputs['RIGHT']
            low = self.inputs['LEFT']

        obs = np.zeros(len(self.inputs))
        if (self.in_epoch(self.t, 'fixation') or
           self.in_epoch(self.t, 'stimulus')):
            obs[self.inputs['FIXATION']] = 1
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
            self.perf, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward,
                                       self.num_tr_perf,
                                       tr_perf)
            self.trial = self._new_trial()
        else:
            self.t += self.dt

        self.store_data(obs, action, reward)
        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8
