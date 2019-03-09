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
import tasktools
import ngym


class ReadySetGo(ngym.ngym):
    def __init__(self, dt=80):
        super().__init__(dt=dt)
        if dt > 80:
            raise ValueError('dt {:0.2f} too large for this task.'.format(dt))
        # Inputs
        self.inputs = tasktools.to_map('FIXATION', 'READY', 'SET')

        # Actions
        self.actions = tasktools.to_map('FIXATE', 'GO')

        # Input noise
        self.sigma = np.sqrt(2*100*0.01)

        # Durations
        self.fixation = 500
        self.gain = 1
        self.ready = 83
        self.set = 83
        max_trial_duration = self.fixation + 2500
        print('max trial duration: ' + str(max_trial_duration) +
              ' (max num. steps: ' + str(max_trial_duration/self.dt) +
              ')')
        # Rewards
        self.R_ABORTED = -1.
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False
        # set action and observation space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.trial = self._new_trial()

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        measure = tasktools.choice(self.rng, [500, 580, 660, 760,
                                              840, 920, 1000])
        gain = 1
        production = measure * gain
        self.tmax = self.fixation + measure + self.set + 2*production

        durations = {
            'fix_grace': (0, 100),
            'fixation':  (0, self.fixation),
            'ready': (self.fixation, self.fixation + self.ready),
            'measure': (self.fixation, self.fixation + measure),
            'set': (self.fixation + measure,
                    self.fixation + measure + self.set),
            'production': (self.fixation + measure + self.set,
                           self.tmax),
            }

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        return {
            'durations': durations,
            'measure': measure,
            'production': production,
            }

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'continue': True}
        reward = 0
        tr_perf = False
        if not self.in_epoch(self.t, 'production'):
            if (action != self.actions['FIXATE'] and
                    not self.in_epoch(self.t, 'fix_grace')):
                info['continue'] = not self.abort
                reward = self.R_ABORTED
        else:
            if action == self.actions['GO']:
                info['continue'] = False  # terminate
                # actual production time
                t_prod = self.t - trial['durations']['measure'][1]
                eps = abs(t_prod - trial['production'])
                eps_threshold = 0.2*trial['production']+25
                if eps > eps_threshold:
                    info['correct'] = False
                    reward = self.R_FAIL
                else:
                    info['correct'] = True
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = min(reward, 0.1)
                    reward *= self.R_CORRECT
                tr_perf = True

                info['t_choice'] = self.t

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        obs = np.zeros(len(self.inputs))
        obs[self.inputs['FIXATION']] = 1  # TODO: this is always on now
        if self.in_epoch(self.t, 'ready'):
            obs[self.inputs['READY']] = 1
        if self.in_epoch(self.t, 'set'):
            obs[self.inputs['SET']] = 1

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['continue'],
                                                self.R_MISS, reward)

        if new_trial:
            info['new_trial'] = True
            info['gt'] = trial['measure']
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward,
                                       self.num_tr_perf, tr_perf)
            self.trial = self._new_trial()
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

        return p_decision >= 0.99 and p_correct >= 0.8
