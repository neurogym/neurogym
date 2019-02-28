#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:31:14 2019

@author: molano

Perceptual decision-making with postdecision wagering, based on

  Representation of confidence associated with a decision by
  neurons in the parietal cortex.
  R. Kiani & M. N. Shadlen, Science 2009.

  http://dx.doi.org/10.1126/science.1169405

"""
from __future__ import division
import ngym
from gym import spaces
import numpy as np

import tasktools


class PDWager(ngym.ngym):
    # Inputs
    inputs = tasktools.to_map('FIXATION', 'LEFT', 'RIGHT', 'SURE')

    # Actions
    actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT',
                               'CHOOSE-SURE')

    # Trial conditions
    wagers = [True, False]
    left_rights = [-1, 1]
    cohs = [0, 3.2, 6.4, 12.8, 25.6, 51.2]

    # Input noise
    sigma = np.sqrt(2*100*0.01)

    # Separate inputs
    N = 100
    Wins = []
    for i in range(3):
        Win = np.zeros((len(inputs), N))
        Win[inputs['FIXATION']] = 1
        Win[inputs['LEFT'], :N//2] = 1
        Win[inputs['RIGHT'], :N//2] = 1
        Win[inputs['SURE'], N//2:] = 1
        Wins.append(Win)
    Win = np.concatenate(Wins, axis=1)

    # Durations
    fixation = 750
    stimulus_min = 100
    stimulus_mean = 180
    stimulus_max = 800
    delay_min = 1200
    delay_mean = 1350
    delay_max = 1800
    sure_min = 500
    sure_mean = 575
    sure_max = 750
    decision = 500
    tmax = fixation + stimulus_min + stimulus_max + delay_max + decision

    # Rewards
    R_ABORTED = -1.
    R_CORRECT = +1.
    R_MISS = 0.
    R_SURE = 0.7*R_CORRECT

    def __init__(self, dt=50):
        # call ngm __init__ function
        super().__init__(dt=dt)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4, ),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.steps_beyond_done = None

        self.trial = self._new_trial(self.rng, self.dt)

    # Input scaling
    def scale(self, coh):
        return (1 + coh/100)/2

    def _new_trial(self, rng, dt, context={}):
        # ---------------------------------------------------------------------
        # Wager or no wager?
        # ---------------------------------------------------------------------

        wager = context.get('wager')
        if wager is None:
            wager = rng.choice(self.wagers)

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        stimulus = context.get('stimulus')
        if stimulus is None:
            stimulus = self.stimulus_min +\
                tasktools.truncated_exponential(rng, dt, self.stimulus_mean,
                                                xmax=self.stimulus_max)

        delay = context.get('delay')
        if delay is None:
            delay = tasktools.truncated_exponential(rng, dt, self.delay_mean,
                                                    xmin=self.delay_min,
                                                    xmax=self.delay_max)

        if wager:
            sure_onset = context.get('sure_onset')
            if sure_onset is None:
                sure_onset =\
                    tasktools.truncated_exponential(rng, dt, self.sure_mean,
                                                    xmin=self.sure_min,
                                                    xmax=self.sure_max)

        durations = {
            'fixation':  (0, self.fixation),
            'stimulus':  (self.fixation, self.fixation + stimulus),
            'delay':     (self.fixation + stimulus,
                          self.fixation + stimulus + delay),
            'decision':  (self.fixation + stimulus + delay, self.tmax),
            'tmax':      self.tmax
            }
        if wager:
            durations['sure'] = (self.fixation + stimulus + sure_onset,
                                 self.tmax)
        time, epochs = tasktools.get_epochs_idx(dt, durations)

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        left_right = context.get('left_right')
        if left_right is None:
            left_right = rng.choice(self.left_rights)

        coh = context.get('coh')
        if coh is None:
            coh = rng.choice(self.cohs)

        return {
            'durations':  durations,
            'time':       time,
            'epochs':     epochs,
            'wager':      wager,
            'left_right': left_right,
            'coh':        coh
            }

    def step(self, action):
        trial = self.trial
        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------

        # epochs = trial['epochs']
        info = {'continue': True}
        reward = 0
        tr_perf = False
        if not self.in_epoch(self.t, 'decision'):
            if action != self.actions['FIXATE']:
                info['continue'] = False
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
            elif action == self.actions['CHOOSE-RIGHT']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'R'
                info['t_choice'] = self.t
                info['correct'] = (trial['left_right'] > 0)
                if info['correct']:
                    reward = self.R_CORRECT
            elif action == self.actions['CHOOSE-SURE']:
                tr_perf = True
                info['continue'] = False
                if trial['wager']:
                    info['choice'] = 'S'
                    info['t_choice'] = self.t
                    reward = self.R_SURE
                else:
                    reward = self.R_ABORTED

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
           self.in_epoch(self.t, 'stimulus') or
                self.in_epoch(self.t, 'delay')):
            obs[self.inputs['FIXATION']] = 1
        if self.in_epoch(self.t, 'stimulus'):
            obs[high] = self.scale(+trial['coh']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
            obs[low] = self.scale(-trial['coh']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
        if trial['wager'] and self.in_epoch(self.t, 'sure'):
            obs[self.inputs['SURE']] = 1

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
                                       self.p_stp, self.num_tr_perf, tr_perf)
            self.trial = self._new_trial(self.rng, self.dt)
        else:
            self.t += 1

        done = False  # TODO: revisit
        return obs, reward, done, info

    def terminate(perf):
        p_answer = perf.n_answer/perf.n_trials
        p_correct = tasktools.divide(perf.n_correct, perf.n_decision)
        p_sure = tasktools.divide(perf.n_sure, perf.n_sure_decision)

        return p_answer >= 0.99 and p_correct >= 0.79 and 0.4 < p_sure <= 0.5
