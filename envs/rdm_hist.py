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

    def __init__(self, dt=100, rep_prob=(.2, .8), block_dur=200):
        super().__init__(dt=dt)
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, ),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.steps_beyond_done = None

        # repeating prob variables
        # prob. of repeating the stimuli in the positions of previous trial
        self.rep_prob = rep_prob
        # position of the first stimulus
        self.left_right_prev_trial = self.rng.choice([0, 1])
        # keeps track of the repeating prob of the current block
        self.curr_block = self.rng.choice([0, 1])
        # duration of block (in number oif trials)
        self.block_dur = block_dur

        self.trial = self._new_trial(self.rng, self.dt)
        print('------------------------')
        print('RDM with history dependencies task')
        print('time step: ' + str(self.dt))
        print('------------------------')

    def _new_trial(self, rng, dt, context={}):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        stimulus = context.get('stimulus')
        if stimulus is None:
            stimulus = tasktools.truncated_exponential(rng, dt,
                                                       self.stimulus_mean,
                                                       xmin=self.stimulus_min,
                                                       xmax=self.stimulus_max)

        durations = {
            'fixation':  (0, self.fixation),
            'stimulus':  (self.fixation, self.fixation + stimulus),
            'decision':  (self.fixation + stimulus,
                          self.fixation + stimulus + self.decision),
            'tmax':      self.tmax
            }
        time, epochs = tasktools.get_epochs_idx(dt, durations)

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        left_right = context.get('left_right')
        if left_right is None:
            if self.left_right_prev_trial == -1:
                probs = (self.rep_prob[self.curr_block],
                         1-self.rep_prob[self.curr_block])
            else:
                probs = (1-self.rep_prob[self.curr_block],
                         self.rep_prob[self.curr_block])
            left_right = rng.choice(self.left_rights,
                                    p=probs)

        self.left_right_prev_trial = left_right

        coh = context.get('coh')
        if coh is None:
            coh = rng.choice(self.cohs)

        return {
            'durations':   durations,
            'time':        time,
            'epochs':      epochs,
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
        epochs = trial['epochs']
        status = {'continue': True}
        status['gt'] = trial['left_right']
        reward = 0
        tr_perf = False
        if self.t-1 not in epochs['decision']:
            if action != self.actions['FIXATE']:
                status['continue'] = False
                reward = self.R_ABORTED
        elif self.t-1 in epochs['decision']:
            if action == self.actions['CHOOSE-LEFT']:
                tr_perf = True
                status['continue'] = False
                status['choice'] = 'L'
                status['t_choice'] = self.t-1
                status['correct'] = (trial['left_right'] < 0)
                if status['correct']:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL
            elif action == self.actions['CHOOSE-RIGHT']:
                tr_perf = True
                status['continue'] = False
                status['choice'] = 'R'
                status['t_choice'] = self.t-1
                status['correct'] = (trial['left_right'] > 0)
                if status['correct']:
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
        if self.t in epochs['fixation'] or self.t in epochs['stimulus']:
            obs[self.inputs['FIXATION']] = 1
        if self.t in epochs['stimulus']:
            obs[high] = self.scale(+trial['coh']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
            obs[low] = self.scale(-trial['coh']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial, self.t, self.perf, self.num_tr, self.num_tr_perf =\
            tasktools.new_trial(self.t, self.tmax, self.dt, status['continue'],
                                self.R_MISS, self.num_tr, self.perf, reward,
                                self.p_stp, self.num_tr_perf, tr_perf)

        if new_trial:
            if self.num_tr % self.block_dur == 0:
                self.curr_block = int(not self.curr_block)
            self.trial = self._new_trial(self.rng, self.dt)

        done = False  # TODO: revisit
        return obs, reward, done, status

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8
